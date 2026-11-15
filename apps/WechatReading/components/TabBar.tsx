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
import { useLocation } from 'react-router-dom';
import { useWechatReadingStore } from '../state';
import { useVisualOrder, useVisualThemeEnabled } from '../../../os/visual/hooks';
import { manifest } from '../manifest';
import { IcTabReading, IcTabBookshelf, IcTabAudiobooks, IcTabMe, IcTabCommunity } from '../res/icons';
import { useWechatReadingGestures } from '../hooks/useWechatReadingGestures';
import { useWechatReadingStrings } from '../hooks/useWechatReadingStrings';

const TabBar: React.FC = () => {
  const visualOrder = useVisualOrder('wechat_reading');
  const visualThemeEnabled = useVisualThemeEnabled('wechat_reading');
  const location = useLocation();
  const audioSubTab = useWechatReadingStore(s => s._temp.audioSubTab);
  const { bindTap } = useWechatReadingGestures();
  const s = useWechatReadingStrings();

  const activeTab =
    location.pathname === '/bookshelf'
      ? 'bookshelf'
      : location.pathname === '/audiobooks'
        ? 'audiobooks'
        : location.pathname === '/me'
          ? 'me'
          : 'reading';

  const tabs = [
    { id: 'reading', label: s.tab_reading },
    { id: 'bookshelf', label: s.tab_bookshelf },
    { id: 'audiobooks', label: s.tab_audiobooks },
    { id: 'me', label: s.tab_me },
  ] as const;

  const getIcon = (id: string, active: boolean) => {
    const color = active ? (visualThemeEnabled ? 'var(--app-primary)' : manifest.theme.colors.primary) : '#999999';
    const size = 24;

    switch (id) {
      case 'reading':
        return <IcTabReading size={size} color={color} fill={active ? color : 'none'} />;
      case 'bookshelf':
        return <IcTabBookshelf size={size} color={color} fill={active ? color : 'none'} />;
      case 'audiobooks':
        return audioSubTab === 'community'
          ? <IcTabCommunity size={size} color={color} fill={active ? color : 'none'} />
          : <IcTabAudiobooks size={size} color={color} fill={active ? color : 'none'} />;
      case 'me':
        return <IcTabMe size={size} color={color} fill={active ? color : 'none'} />;
      default:
        return null;
    }
  };

  const getLabel = (id: string, defaultLabel: string) => {
    if (id === 'audiobooks' && audioSubTab === 'community') {
      return s.tab_book_friends;
    }

    return defaultLabel;
  };

  return (
    <div data-visual-group="wechat_reading.primary-nav" className="absolute bottom-0 left-0 right-0 h-14 bg-app-surface border-t border-(--app-c-tw-border-slate-100) flex items-center justify-around pb-safe z-50">
      {tabs.map(tab => {
        const isActive = activeTab === tab.id;
        const tapBinding =
          tab.id === 'reading'
            ? bindTap<HTMLButtonElement>('tab.reading')
            : tab.id === 'bookshelf'
              ? bindTap<HTMLButtonElement>('tab.bookshelf')
              : tab.id === 'audiobooks'
                ? bindTap<HTMLButtonElement>('tab.audiobooks', { params: { sub: audioSubTab } })
                : bindTap<HTMLButtonElement>('tab.me');

        return (
          <button
            key={tab.id}
            data-visual-item={tab.id} style={visualOrder(tab.id)}
            {...tapBinding}
            className="flex flex-col items-center justify-center flex-1 h-full active:opacity-70"
          >
            <div className="mb-0.5">{getIcon(tab.id, isActive)}</div>
            <span className={`text-(--app-tab-bar-label-size) ${isActive ? 'text-app-primary' : 'text-(--app-c-tab-bar-text-9999)'}`}>
              {getLabel(tab.id, tab.label)}
            </span>
          </button>
        );
      })}
    </div>
  );
};

export default TabBar;
