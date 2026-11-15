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

import React from 'react';
import { useWechatStrings } from '../hooks/useWechatStrings';
import { IcTabWechat, IcTabContacts, IcTabDiscover, IcTabMe } from '../res/icons';
import { dimens } from '../res/dimens';
import { useLocation } from 'react-router-dom';
import { useWechatGestures } from '../hooks/useWechatGestures';
import { useVisualOrder } from '../../../os/visual/hooks';
const TabBar: React.FC = () => {
  const visualOrder = useVisualOrder('wechat');
  const t = useWechatStrings();
  const location = useLocation();
  const { bindTap, go } = useWechatGestures();
  const currentPath = location.pathname;

  const isActive = (path: string) => currentPath === path;
  const showTabs = ['/', '/contacts', '/discover', '/me'].includes(currentPath);

  if (!showTabs) return null;

  const activeColor = "text-app-primary";
  const inactiveColor = "text-(--app-c-common-text-primary)";

  const handleTabClick = (path: string, transitionId: 'tab.wechat' | 'tab.contacts' | 'tab.discover' | 'tab.me') => {
    if (currentPath === path) return;
    go(transitionId);
  };

  return (
    <div data-visual-group="wechat.primary-nav" className="h-(--app-item-height-65) bg-(--app-c-chat-input-bar-bg) border-t border-(--app-c-tw-border-gray-300) flex justify-around items-start w-full pt-2 absolute bottom-0 left-0 right-0 z-[90]">
      <div
        data-visual-item="wechat" style={visualOrder('wechat')}
        {...bindTap<HTMLDivElement>('tab.wechat', { onTrigger: () => handleTabClick('/', 'tab.wechat') })}
        className={`flex flex-col items-center justify-start w-full space-y-[2px] cursor-pointer`}
      >
        <div className={`relative ${isActive('/') ? activeColor : inactiveColor}`}>
          <IcTabWechat size={dimens.icSizeTab} fill={isActive('/') ? "currentColor" : "none"} strokeWidth={isActive('/') ? 0 : 1.5} />
        </div>
        <span className={`text-(--app-hint-text-size-10) ${isActive('/') ? activeColor : 'text-(--app-c-tw-text-gray-500)'}`}>{t.tab_wechat}</span>
      </div>

      <div
        data-visual-item="contacts" style={visualOrder('contacts')}
        {...bindTap<HTMLDivElement>('tab.contacts', { onTrigger: () => handleTabClick('/contacts', 'tab.contacts') })}
        className={`flex flex-col items-center justify-start w-full space-y-[2px] cursor-pointer`}
      >
        <div className={`${isActive('/contacts') ? activeColor : inactiveColor}`}>
          <IcTabContacts size={dimens.icSizeTab} fill={isActive('/contacts') ? "currentColor" : "none"} strokeWidth={isActive('/contacts') ? 0 : 1.5} />
        </div>
        <span className={`text-(--app-hint-text-size-10) ${isActive('/contacts') ? activeColor : 'text-(--app-c-tw-text-gray-500)'}`}>{t.tab_contacts}</span>
      </div>

      <div
        data-visual-item="discover" style={visualOrder('discover')}
        {...bindTap<HTMLDivElement>('tab.discover', { onTrigger: () => handleTabClick('/discover', 'tab.discover') })}
        className={`flex flex-col items-center justify-start w-full space-y-[2px] cursor-pointer`}
      >
        <div className={`${isActive('/discover') ? activeColor : inactiveColor}`}>
          <IcTabDiscover size={dimens.icSizeTab} fill={isActive('/discover') ? "currentColor" : "none"} strokeWidth={isActive('/discover') ? 0 : 1.5} />
        </div>
        <span className={`text-(--app-hint-text-size-10) ${isActive('/discover') ? activeColor : 'text-(--app-c-tw-text-gray-500)'}`}>{t.tab_discover}</span>
      </div>

      <div
        data-visual-item="me" style={visualOrder('me')}
        {...bindTap<HTMLDivElement>('tab.me', { onTrigger: () => handleTabClick('/me', 'tab.me') })}
        className={`flex flex-col items-center justify-start w-full space-y-[2px] cursor-pointer`}
      >
        <div className={`${isActive('/me') ? activeColor : inactiveColor}`}>
          <IcTabMe size={dimens.icSizeTab} fill={isActive('/me') ? "currentColor" : "none"} strokeWidth={isActive('/me') ? 0 : 1.5} />
        </div>
        <span className={`text-(--app-hint-text-size-10) ${isActive('/me') ? activeColor : 'text-(--app-c-tw-text-gray-500)'}`}>{t.tab_me}</span>
      </div>
    </div>
  );
};

export default TabBar;
