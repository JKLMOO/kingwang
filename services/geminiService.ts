import { GoogleGenAI, Type } from "@google/genai";
import { DetectionResponse } from "../types";

const API_KEY = process.env.API_KEY;
if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const model = ai.models;

export async function fetchHerbDetails(herbName: string): Promise<{ category: string; features: string }> {
  const prompt = `你是一位中药植物视觉专家。请根据【${herbName}】，输出以下信息：
1. 类别： (请回答：草本、灌木 或 木本/树木)
2. 视觉特征： (请以列表形式输出茎、叶、花、果的肉眼可见形态)`;

  const response = await model.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
  });

  const text = response.text;
  
  let category = '';
  let features = '';

  const lines = text.split('\n');
  const categoryLineIndex = lines.findIndex(line => line.includes('类别：'));
  const featuresLineIndex = lines.findIndex(line => line.includes('视觉特征：'));

  if (categoryLineIndex > -1) {
    // Define a search area for the category, which is between the "类别：" line
    // and the "视觉特征：" line (or the end of the text).
    const searchEndIndex = featuresLineIndex > -1 ? featuresLineIndex : lines.length;
    const searchBlock = lines.slice(categoryLineIndex, searchEndIndex).join('\n');
    
    if (searchBlock.includes('草本')) category = '草本';
    else if (searchBlock.includes('木本') || searchBlock.includes('树木')) category = '木本';
    else if (searchBlock.includes('灌木')) category = '灌木';
  }

  if (featuresLineIndex > -1) {
    // The features are everything after the "视觉特征：" line.
    features = lines.slice(featuresLineIndex + 1).join('\n').trim();
  } else if (category) {
    // Fallback: If no features header, but we have a category, assume features start
    // after the line where the category value was found.
    const categoryValueLineIndex = lines.findIndex(line => line.includes(category));
    if (categoryValueLineIndex > -1) {
      features = lines.slice(categoryValueLineIndex + 1).join('\n').trim();
    }
  }

  if (!category || !features) {
    console.error("Failed to parse herb details from AI response. Raw text was:", `\n---\n${text}\n---`);
    throw new Error('Failed to parse herb details from AI response.');
  }

  return { category, features };
}

const DETECTION_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    found: { type: Type.BOOLEAN },
    summary: { type: Type.STRING },
    box: {
      type: Type.OBJECT,
      properties: {
        x_min: { type: Type.NUMBER },
        y_min: { type: Type.NUMBER },
        x_max: { type: Type.NUMBER },
        y_max: { type: Type.NUMBER },
      },
      required: ['x_min', 'y_min', 'x_max', 'y_max'],
    },
  },
  required: ['found', 'summary', 'box'],
};

export async function analyzeImage(
  base64Image: string,
  herbFeatures: string
): Promise<DetectionResponse> {
    const prompt = `这是一张摄像头实时拍摄的植物图片。请根据以下已知的目标植物视觉特征，判断图片中是否存在该植物。
---
已知视觉特征:
${herbFeatures}
---
请严格按照 JSON 格式输出结果，包含是否找到("found")，一个总结("summary")，以及一个边界框("box")。总结中必须包含置信度百分比。边界框坐标必须是0-1000范围内的归一化整数。如果未找到，返回一个覆盖全图的默认框。`;

    const imagePart = {
        inlineData: {
            data: base64Image,
            mimeType: 'image/jpeg',
        },
    };

    const textPart = { text: prompt };

    const response = await model.generateContent({
        model: 'gemini-2.5-pro',
        contents: { parts: [imagePart, textPart] },
        config: {
            responseMimeType: "application/json",
            responseSchema: DETECTION_SCHEMA,
        }
    });

    try {
        const jsonString = response.text;
        const result: DetectionResponse = JSON.parse(jsonString);
        return result;
    } catch (e) {
        console.error("Failed to parse JSON response from Gemini:", e);
        throw new Error("AI response was not valid JSON.");
    }
}