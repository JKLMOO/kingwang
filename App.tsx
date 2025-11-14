
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { AppStage, Box, DetectionResponse, TfjsPrediction } from './types';
import { fetchHerbDetails, analyzeImage } from './services/geminiService';

// Declare global variables from CDN scripts
declare const Tone: any;
declare const tf: any;
declare const cocoSsd: any;

const API_CALL_INTERVAL = 2500; // 2.5 seconds

const App: React.FC = () => {
    const [appStage, setAppStage] = useState<AppStage>(AppStage.INPUT);
    const [herbName, setHerbName] = useState<string>('');
    const [herbFeatures, setHerbFeatures] = useState<string>('');
    const [herbCategory, setHerbCategory] = useState<string>('');

    const [detectionBox, setDetectionBox] = useState<Box | null>(null);
    const [confidence, setConfidence] = useState<number>(0);
    const [aiSummary, setAiSummary] = useState<string>('');
    
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [statusText, setStatusText] = useState<string>('输入目标中药名称开始。');

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const tfjsModelRef = useRef<any>(null);
    const requestAnimationRef = useRef<number>(0);
    const lastApiCallTimestampRef = useRef<number>(0);
    const toneSynthRef = useRef<any>(null);

    useEffect(() => {
        // Initialize Tone.js synth
        toneSynthRef.current = new Tone.Synth().toDestination();

        // Load TensorFlow.js model
        const loadModel = async () => {
            try {
                tfjsModelRef.current = await cocoSsd.load();
                console.log("COCO-SSD model loaded.");
            } catch (err) {
                console.error("Failed to load TFJS model", err);
                setError("无法加载本地 AI 模型。请刷新页面重试。");
                setAppStage(AppStage.ERROR);
            }
        };
        loadModel();
    }, []);

    const handleFetchFeatures = async () => {
        if (!herbName) {
            setError('请输入中药名称。');
            return;
        }
        setIsLoading(true);
        setError(null);
        setAppStage(AppStage.FETCHING_FEATURES);
        setStatusText(`正在检索【${herbName}】的特征...`);

        try {
            // Check localStorage cache first
            const cachedData = localStorage.getItem(`herbvisionpro-${herbName}`);
            if (cachedData) {
                const { category, features } = JSON.parse(cachedData);
                setHerbCategory(category);
                setHerbFeatures(features);
                setStatusText(`已从缓存加载特征。准备启动摄像头...`);
            } else {
                const { category, features } = await fetchHerbDetails(herbName);
                setHerbCategory(category);
                setHerbFeatures(features);
                localStorage.setItem(`herbvisionpro-${herbName}`, JSON.stringify({ category, features }));
                setStatusText(`特征检索成功！准备启动摄像头...`);
            }
            setAppStage(AppStage.READY_TO_STREAM);
        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : '未知错误');
            setAppStage(AppStage.ERROR);
            setStatusText(`特征检索失败。`);
        } finally {
            setIsLoading(false);
        }
    };
    
    const startStreaming = async () => {
        if (appStage !== AppStage.READY_TO_STREAM) return;

        setAppStage(AppStage.STREAMING);
        setStatusText('正在启动摄像头...');
        setError(null);

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setError("您的浏览器不支持摄像头访问功能。");
            setAppStage(AppStage.ERROR);
            return;
        }

        const setupStream = (stream: MediaStream) => {
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    setStatusText('摄像头已启动，开始实时扫描...');
                    runDetectionLoop();
                };
            }
        };

        const videoConstraints = [
            { video: { facingMode: 'environment' } }, // 1. Try rear camera
            { video: true } // 2. Fallback to any camera
        ];

        let stream: MediaStream | null = null;
        let lastError: Error | null = null;

        for (const constraints of videoConstraints) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (stream) {
                    lastError = null;
                    break; // Success!
                }
            } catch (err) {
                console.warn(`Failed to get camera with constraints ${JSON.stringify(constraints)}`, err);
                if (err instanceof Error) {
                    lastError = err;
                }
            }
        }
        
        if (stream) {
            setupStream(stream);
        } else {
            console.error("Could not start camera stream:", lastError);
            let errorMessage = '无法访问摄像头。';
            if (lastError) {
                switch (lastError.name) {
                    case 'NotFoundError':
                    case 'DevicesNotFoundError':
                        errorMessage = '未找到可用的摄像头设备。';
                        break;
                    case 'NotAllowedError':
                    case 'PermissionDeniedError':
                        errorMessage = '摄像头权限被拒绝。请在浏览器设置中允许访问。';
                        break;
                    case 'NotReadableError':
                    case 'TrackStartError':
                         errorMessage = '摄像头已被其他应用占用或硬件错误。';
                         break;
                    default:
                        errorMessage = '启动摄像头时发生未知错误。';
                }
            }
            setError(errorMessage);
            setAppStage(AppStage.ERROR);
        }
    };
    
    const stopStreaming = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        if (requestAnimationRef.current) {
            cancelAnimationFrame(requestAnimationRef.current);
        }
        setAppStage(AppStage.INPUT);
        setHerbName('');
        setHerbFeatures('');
        setHerbCategory('');
        setDetectionBox(null);
        setConfidence(0);
        setAiSummary('');
        setStatusText('输入目标中药名称开始。');
    };

    const captureAndAnalyze = useCallback(async () => {
        if (!videoRef.current || !herbFeatures) return;
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoRef.current.videoWidth;
        tempCanvas.height = videoRef.current.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
        const base64Image = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        setStatusText('检测到潜在目标，正在上传云端进行高精度比对...');

        try {
            const result: DetectionResponse = await analyzeImage(base64Image, herbFeatures);
            
            setAiSummary(result.summary);
            
            if (result.found) {
                setDetectionBox(result.box);
                const confidenceMatch = result.summary.match(/置信度：(\d+)%/);
                const newConfidence = confidenceMatch ? parseInt(confidenceMatch[1], 10) : 0;
                setConfidence(newConfidence);
            } else {
                setConfidence(0);
                setDetectionBox(null);
            }
            setStatusText(`比对完成: ${result.summary}`);
        } catch (err) {
            console.error('Analysis failed', err);
            setStatusText('云端比对失败，请检查网络连接。');
            setConfidence(0);
            setDetectionBox(null);
        }
    }, [herbFeatures]);

    const runDetectionLoop = useCallback(async () => {
        if (!tfjsModelRef.current || !videoRef.current || videoRef.current.readyState < 3) {
            requestAnimationRef.current = requestAnimationFrame(runDetectionLoop);
            return;
        }

        const predictions: TfjsPrediction[] = await tfjsModelRef.current.detect(videoRef.current);
        
        let trigger = false;
        const triggerClasses = {
            '草本': ['potted plant', 'flower'],
            '木本': ['tree'],
            '灌木': ['potted plant', 'flower', 'tree'], // Broader category
            'default': ['plant', 'potted plant', 'flower']
        };

        const targetClasses = triggerClasses[herbCategory as keyof typeof triggerClasses] || triggerClasses.default;

        for (const prediction of predictions) {
            if (targetClasses.includes(prediction.class) && prediction.score > 0.5) {
                trigger = true;
                break;
            }
        }
        
        const now = Date.now();
        if (trigger && (now - lastApiCallTimestampRef.current > API_CALL_INTERVAL)) {
            lastApiCallTimestampRef.current = now;
            captureAndAnalyze();
        }

        requestAnimationRef.current = requestAnimationFrame(runDetectionLoop);
    }, [herbCategory, captureAndAnalyze]);
    
    // Effect for Audio/Visual Feedback
    useEffect(() => {
        if (appStage !== AppStage.STREAMING) return;
        
        // Audio Feedback
        if (confidence > 0) {
            // Tone.js
            if (toneSynthRef.current) {
                const freq = 200 + confidence * 6; // Pitch scales with confidence
                toneSynthRef.current.triggerAttackRelease(freq, "8n");
            }
            // TTS
            if (aiSummary && 'speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(aiSummary);
                utterance.lang = 'zh-CN';
                window.speechSynthesis.speak(utterance);
            }
        }

        // Visual Feedback (Canvas)
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (detectionBox && confidence > 10) {
            const scaleX = canvas.width / 1000;
            const scaleY = canvas.height / 1000;

            const x = detectionBox.x_min * scaleX;
            const y = detectionBox.y_min * scaleY;
            const width = (detectionBox.x_max - detectionBox.x_min) * scaleX;
            const height = (detectionBox.y_max - detectionBox.y_min) * scaleY;

            const opacity = Math.min(1, confidence / 100);
            const lineWidth = 2 + (confidence / 100) * 8;

            ctx.strokeStyle = `rgba(50, 255, 50, ${opacity})`;
            ctx.lineWidth = lineWidth;
            ctx.shadowColor = `rgba(50, 255, 50, 1)`;
            ctx.shadowBlur = 20;
            ctx.strokeRect(x, y, width, height);

             // Draw confidence text
             ctx.fillStyle = `rgba(50, 255, 50, ${opacity})`;
             ctx.font = 'bold 24px sans-serif';
             ctx.textAlign = 'center';
             ctx.fillText(`${confidence}%`, x + width / 2, y > 30 ? y - 10 : y + height + 25);
        }
    }, [confidence, detectionBox, aiSummary, appStage]);


    return (
        <div className="min-h-screen bg-gray-900 text-green-300 font-mono flex flex-col items-center p-4">
            <header className="w-full max-w-4xl text-center mb-4 border-b-2 border-green-500 pb-2">
                <h1 className="text-3xl md:text-4xl font-bold tracking-widest text-shadow-glow">
                    🌿 HerbVision Pro
                </h1>
                <p className="text-sm text-green-400">“寻宝探测器”式中药识别应用</p>
            </header>

            <main className="w-full max-w-4xl flex-grow flex flex-col items-center">
                {appStage < AppStage.STREAMING && (
                    <div className="w-full p-6 bg-gray-800 border-2 border-green-700 rounded-lg shadow-lg flex flex-col gap-4">
                        <div className="flex flex-col md:flex-row gap-4">
                             <input
                                type="text"
                                value={herbName}
                                onChange={(e) => setHerbName(e.target.value)}
                                placeholder="例如：人参"
                                disabled={isLoading}
                                className="flex-grow bg-gray-900 border border-green-500 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-400 disabled:opacity-50"
                            />
                            <button
                                onClick={handleFetchFeatures}
                                disabled={isLoading || !herbName}
                                className="bg-green-600 hover:bg-green-500 text-gray-900 font-bold py-2 px-4 rounded transition-colors duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed"
                            >
                                {isLoading ? '检索中...' : '特征检索'}
                            </button>
                        </div>
                        {herbFeatures && appStage === AppStage.READY_TO_STREAM && (
                            <div className="mt-4 p-4 bg-gray-900/50 rounded border border-green-800">
                                <h3 className="font-bold text-lg">【{herbName}】 - {herbCategory}</h3>
                                <pre className="whitespace-pre-wrap text-sm text-green-200 mt-2 max-h-40 overflow-y-auto">{herbFeatures}</pre>
                                <button
                                    onClick={startStreaming}
                                    className="mt-4 w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded transition-colors duration-300"
                                >
                                    启动实时比对
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {appStage === AppStage.STREAMING && (
                    <div className="relative w-full aspect-video max-w-4xl border-4 border-green-700 rounded-lg overflow-hidden shadow-2xl shadow-green-500/20">
                        <video ref={videoRef} className="w-full h-full object-cover" playsInline />
                        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
                        <div className="absolute top-0 left-0 w-full h-full pointer-events-none bg-grid-pattern opacity-10"></div>
                        <div className="absolute top-0 left-0 w-full h-full pointer-events-none animate-scanline"></div>
                    </div>
                )}

                <div className="w-full max-w-4xl mt-4 p-3 bg-black/50 border border-green-800 rounded text-center">
                    <p className="font-bold">状态: <span className="text-yellow-300">{statusText}</span></p>
                    {error && <p className="text-red-400 mt-1">错误: {error}</p>}
                </div>
                
                 {appStage === AppStage.STREAMING && (
                     <button
                        onClick={stopStreaming}
                        className="mt-4 bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-6 rounded transition-colors duration-300"
                    >
                        停止 & 重置
                    </button>
                 )}
            </main>

            <style>{`
                .text-shadow-glow {
                    text-shadow: 0 0 5px #34d399, 0 0 10px #34d399;
                }
                .bg-grid-pattern {
                    background-image: linear-gradient(rgba(52, 211, 153, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(52, 211, 153, 0.2) 1px, transparent 1px);
                    background-size: 20px 20px;
                }
                @keyframes scanline {
                    0% {
                        transform: translateY(-10%);
                        opacity: 0.1;
                    }
                    100% {
                        transform: translateY(110%);
                        opacity: 0.1;
                    }
                }
                .animate-scanline::after {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(to bottom, rgba(110, 231, 183, 0), rgba(52, 211, 153, 0.5), rgba(110, 231, 183, 0));
                    animation: scanline 4s linear infinite;
                }
            `}</style>
        </div>
    );
};

export default App;
