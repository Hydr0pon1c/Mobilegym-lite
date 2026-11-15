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

import defaults from './defaults.json';
import { localizeKnownExternalAssets } from '../../../os/utils/cdn';
import type { XConversation, XSettings } from '../types';

// 重新导出类型，保持历史 import 兼容
export * from '../types';

const localizedDefaults = localizeKnownExternalAssets(defaults) as any;

export const currentUser = localizedDefaults.user;
// 数据层 contract: 所有 id 已 case-sensitive 唯一, 不再做 .toLowerCase() 归一。
export const defaultFollowedUserIds = (localizedDefaults.user?.followedUserIds ?? []) as string[];
export const defaultFollowerUserIds = (localizedDefaults.user?.followerUserIds ?? []) as string[];
export const trends = localizedDefaults.trends as any[];
export const notifications = localizedDefaults.notifications as any[];
export const conversations = localizedDefaults.conversations as XConversation[];
export const searchHistory = localizedDefaults.searchHistory as any[];
export const xSettings = localizedDefaults.settings as XSettings;

export const X_CONFIG = {
  user: currentUser,
  posts: localizedDefaults.posts ?? {},
  trends,
  notifications,
  conversations,
  recentSearches: searchHistory,
  settings: xSettings,
};
