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

import React, { useState } from 'react';
import type { Alarm } from '../types';
import { strings } from '../res/strings';
import { stringsEn } from '../res/strings.en';
import { useAppStrings } from '../../../os/useAppStrings';
import { useClockGestures } from '../hooks/useClockGestures';
import { CollapsingToolbar, CollapsingLargeTitle, TOOLBAR_SPACER_HEIGHT } from '../../../os/components/CollapsingToolbar';
import { IcAlarm, IcClose, IcCheck, IcList, IcDelete, IcAdd } from '../res/icons';
import { Switch } from '../components/Switch';
import { formatAlarmTime, getRepeatLabel, getNextAlarmText } from '../utils';

const AlarmItem: React.FC<{
  alarm: Alarm;
  selectionMode: boolean;
  selected: boolean;
  onToggle: () => void;
  onSelectToggle: () => void;
  onSelectionStart: () => void;
  onEdit?: () => void;
}> = ({ alarm, selectionMode, selected, onToggle, onSelectToggle, onSelectionStart, onEdit }) => {
  const s = useAppStrings(strings, stringsEn);
  const { bindLongPress } = useClockGestures();

  return (
    <div
      className={`bg-app-surface rounded-2xl px-4 py-4 flex items-center justify-between ${selectionMode && selected ? 'bg-blue-50' : ''}`}
      {...bindLongPress({ kind: 'action', id: 'alarm.item.selection.enter' }, { duration: 450, onTrigger: () => onSelectionStart() })}
      onClick={() => { if (selectionMode) onSelectToggle(); }}
      onContextMenu={(e) => {
        if (!selectionMode) {
          e.preventDefault();
          onSelectionStart();
        }
      }}
    >
      <button
        type="button"
        className="flex flex-col text-left flex-1 min-w-0"
        onClick={() => { if (!selectionMode && onEdit) onEdit(); }}
      >
        <span className="text-[26px] font-medium text-black">{formatAlarmTime(alarm.hour, alarm.minute)}</span>
        <span className="text-[15px] text-gray-400">
          {getRepeatLabel(alarm.repeat, s)}
          {alarm.note ? `｜${alarm.note}` : ''}
        </span>
      </button>
      {selectionMode ? (
        <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${selected ? 'bg-app-primary text-white' : 'border border-gray-300 text-transparent'}`}>
          <IcCheck size={16} />
        </div>
      ) : (
        <div onClick={e => e.stopPropagation()}>
          <Switch value={alarm.enabled} onChange={onToggle} />
        </div>
      )}
    </div>
  );
};

export const AlarmPage: React.FC<{
  alarms: Alarm[];
  selectionMode: boolean;
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
  onOpenAdd: () => void;
  onEdit: (alarm: Alarm) => void;
  onSelectToggle: (id: string) => void;
  onSelectionStart: (id: string) => void;
  onExitSelection: () => void;
  onToggleSelectAll: () => void;
  onDeleteSelected: () => void;
}> = ({
  alarms,
  selectionMode,
  selectedIds,
  onToggle,
  onOpenAdd,
  onEdit,
  onSelectToggle,
  onSelectionStart,
  onExitSelection,
  onToggleSelectAll,
  onDeleteSelected,
}) => {
  const s = useAppStrings(strings, stringsEn);
  const { bindTap } = useClockGestures();
  const [scrollTop, setScrollTop] = useState(0);
  const hasAlarms = alarms.length > 0;
  const allOff = hasAlarms && alarms.every(alarm => !alarm.enabled);
  const nextAlarmText = allOff ? s.alarm_all_off : getNextAlarmText(alarms, s.alarm_no_alarms, s.alarm_rings_in_hours_infix, s.alarm_rings_in_minutes_suffix);
  const isAllSelected = hasAlarms && selectedIds.size === alarms.length;

  const displayTitle = selectionMode ? `${s.selected_count_prefix}${selectedIds.size}${s.selected_count_suffix}` : s.alarm_tab;

  return (
    <div className="h-full w-full bg-app-bg relative flex flex-col">
      <CollapsingToolbar
        title={displayTitle}
        scrollTop={scrollTop}
        alwaysShowSmallTitle={selectionMode}
        bgClass="bg-app-bg"
        leftContent={selectionMode ? (
          <button className="w-10 h-10 flex items-center justify-center -ml-2 active:opacity-60" onClick={onExitSelection}>
            <IcClose size={28} className="text-app-text" />
          </button>
        ) : undefined}
        rightContent={selectionMode ? (
          <div className="flex items-center gap-1">
            <button
              aria-label={s.toolbar_select_all}
              className="w-10 h-10 flex items-center justify-center active:opacity-60"
              {...bindTap<HTMLButtonElement>(
                { kind: 'action', id: 'alarm.selection.toggleAll' },
                { onTrigger: onToggleSelectAll },
              )}
            >
              <IcList size={28} className={isAllSelected ? 'text-app-primary' : 'text-gray-400'} />
            </button>
            <button
              aria-label={s.toolbar_delete}
              className="w-10 h-10 flex items-center justify-center text-red-500 active:opacity-60 disabled:opacity-30"
              disabled={selectedIds.size === 0}
              {...bindTap<HTMLButtonElement>(
                { kind: 'action', id: 'alarm.selection.delete' },
                { onTrigger: onDeleteSelected },
              )}
            >
              <IcDelete size={24} />
            </button>
          </div>
        ) : (
          <button
            aria-label={s.toolbar_add}
            className="w-10 h-10 flex items-center justify-center active:opacity-60"
            {...bindTap<HTMLButtonElement>(
              { kind: 'action', id: 'alarm.editor.open' },
              { onTrigger: onOpenAdd },
            )}
          >
            <IcAdd size={28} className="text-app-text" />
          </button>
        )}
      />

      <div
        className="h-full overflow-y-auto no-scrollbar"
        onScroll={e => setScrollTop((e.target as HTMLDivElement).scrollTop)}
      >
        <div style={{ height: TOOLBAR_SPACER_HEIGHT }} />
        <CollapsingLargeTitle title={displayTitle} scrollTop={scrollTop} />
        {hasAlarms ? (
          <>
            <div className="min-h-[140px] mb-8 flex items-center justify-center px-6">
              <div className="text-[26px] font-medium text-black text-center">{nextAlarmText}</div>
            </div>
            <div className="px-4 flex flex-col gap-3 pb-32">
              {alarms.map(alarm => (
                <AlarmItem
                  key={alarm.id}
                  alarm={alarm}
                  selectionMode={selectionMode}
                  selected={selectedIds.has(alarm.id)}
                  onToggle={() => onToggle(alarm.id)}
                  onSelectToggle={() => onSelectToggle(alarm.id)}
                  onSelectionStart={() => onSelectionStart(alarm.id)}
                  onEdit={() => onEdit(alarm)}
                />
              ))}
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center text-gray-400 mt-32">
            <div className="w-14 h-14 rounded-2xl bg-[#edf2f7] flex items-center justify-center mb-3">
              <IcAlarm size={26} className="text-blue-400" />
            </div>
            <div className="text-[16px]">{s.alarm_no_alarm_set}</div>
          </div>
        )}
      </div>

    </div>
  );
};
