/*
 * Copyright 2026 OPPO
 *
 * Modified by OPPO for this release.
 * OPPO copyright applies only to the modifications.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { benchmarkTime } from './benchmarkTime';

const AI_BASE_URL = import.meta.env.VITE_AI_BASE_URL || '';
const AI_MODEL = import.meta.env.VITE_AI_MODEL || 'qwen-flash';
const AI_API_KEY = import.meta.env.VITE_AI_API_KEY || '';
const benchmarkTimeMs = benchmarkTime(typeof window === 'undefined' ? '' : window.location.search);

export const SIMULATOR_CONFIG = {
  // ======== 框架渲染参数 ========
  framework: {
    screenHeight: 2400,
    screenWidth: 1080,
    dpr: 3,

    viewportWidth: 360,
    viewportHeight: 800,

    statusBarHeight: 40,
    bottomGestureHeight: 16,
    keyboardHeight: 320,

    edgeGestureWidth: 20,
    swipeThreshold: 60,
    gestureBarWidth: 120,
    gestureBarHeight: 4.5,
    homeSwipeThreshold: 20,
    recentsHoldDuration: 350,
    gestureProgressDivisor: 100,
    gestureCancelThreshold: 40,
    gestureProgressScale: 0.3,

    backIndicatorSize: 48,
    backIndicatorOpacity: 0.2,
    gestureBarOpacityLight: 0.6,
    gestureBarOpacityDark: 0.3,

    launcherPaddingTop: 64,
    clockFontSize: 84,
    appGridColumns: 4,
    appIconSize: 60,
    appIconBorderRadius: 12,
    dockBorderRadius: 20,

    recentsCardWidth: 200,
    recentsCardHeight: 434,
    recentsScrollContainerHeight: 500,
    recentsSwipeThreshold: 100,
    recentsBackgroundOpacity: 0.1,
    recentsCardGap: 24,
    recentsCardBorderRadius: 24,
    recentsOpacityDivisor: 150,
    recentsTopPadding: 96,
    recentsAppPreviewWidth: 390,
    recentsAppPreviewHeight: 844,

    zIndexStatusBar: 1000,
    zIndexRecents: 200,
    zIndexRecentsBlur: 200,
    zIndexRecentsCards: 205,
    zIndexRecentsChrome: 210,
    zIndexEdgeGestures: 2500,
    zIndexGestureBar: 3300,
    zIndexKeyboard: 3200,
    zIndexSystemShade: 5000,
    zIndexApp: 50,

    transitionDuration: 200,
    pageTransitionDuration: 300,
    statusBarColorCheckInterval: 300,
  },

  // ======== 行为配置 ========
  time: {
    mode: (benchmarkTimeMs === undefined ? 'real' : 'simulated') as 'real' | 'simulated',
    simulatedTime: benchmarkTimeMs ?? '2026-01-04 20:00:00',
    flowing: benchmarkTimeMs === undefined,
    speed: 1,
  },

  location: {
    mode: 'simulated' as 'real' | 'simulated',
    simulatedLocation: { latitude: 39.9794688, longitude: 116.3323982 } as
      | string
      | { latitude: number; longitude: number; accuracy?: number },
  },

  ai: {
    enabled: Boolean(AI_BASE_URL),
    baseUrl: AI_BASE_URL,
    model: AI_MODEL,
    apiKey: AI_API_KEY,
    temperature: 0.7,
    replyDelay: 500,
    maxContextMessages: 15,
    reasoningEffort: 'none' as 'none' | 'low' | 'medium' | 'high',
  },

  display: {
    scale: 1,
    themeColor: '#1a1a1a',
  },

  intent: {
    chooserEnabled: true,
  },
} as const;
