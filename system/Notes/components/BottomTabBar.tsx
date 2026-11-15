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

import { Feature, FeatureSlot } from '@/os/featureAvailability';
import React from 'react';
import { SIMULATOR_CONFIG } from '@/os/data';
const { bottomGestureHeight } = SIMULATOR_CONFIG.framework;
import { colors } from '../res/colors';
import { IcTabNotes } from '../res/icons';
import { useAppStrings } from '../../../os/useAppStrings';
import { strings } from '../res/strings';
import { stringsEn } from '../res/strings.en';
import { useNotesGestures } from '../hooks/useNotesGestures';
export type NotesTab = 'notes' | 'todo';

export const BottomTabBar: React.FC<{ active: NotesTab }> = ({ active }) => {
  const { go } = useNotesGestures();
  const bottomPad = bottomGestureHeight;
  const s = useAppStrings(strings, stringsEn);

  const Tab: React.FC<{
    tab: NotesTab;
    label: string;
    onClick: () => void;
  }> = ({ tab, label, onClick }) => {
    const isActive = tab === active;
    const labelColor = isActive ? '#000' : colors.text_secondary;
    return (
      <button
        onClick={onClick}
        className="flex flex-col items-center justify-center gap-1 flex-1 h-full active:opacity-70"
        aria-current={isActive ? 'page' : undefined}
      >
        <IcTabNotes tab={tab} active={isActive} size={22} />
        <div className={['text-[12px]', isActive ? 'font-medium' : 'font-normal'].join(' ')} style={{ color: labelColor }}>
          {label}
        </div>
      </button>
    );
  };

  return (
    <div className="absolute left-0 right-0 bottom-0 z-40 bg-white/90 backdrop-blur-xl border-t border-black/5">
      <div className="flex items-center justify-around h-[64px]" style={{ paddingBottom: bottomPad }}>
        <Tab
          tab="notes"
          label={s.notes}
          onClick={() => go('tab.notes')}
        />
        <Feature id="notes.todo" placeholder={
          <FeatureSlot id="notes.todo" className="flex-1 h-full" />
        }><Tab
          tab="todo"
          label={s.todo}
          onClick={() => go('tab.todo')}
        /></Feature>
      </div>
    </div>
  );
};

export default BottomTabBar;
