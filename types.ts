
export enum AppStage {
  INPUT,
  FETCHING_FEATURES,
  READY_TO_STREAM,
  STREAMING,
  ERROR,
}

export interface Box {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface DetectionResponse {
  found: boolean;
  summary: string;
  box: Box;
}

// For coco-ssd model predictions
export interface TfjsPrediction {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}
