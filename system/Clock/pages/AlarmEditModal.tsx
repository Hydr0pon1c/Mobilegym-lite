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
import { getDate } from '../../../os/TimeService';
import { WheelColumn } from '../components/WheelColumn';
import { formatCountdownText, getNextTrigger } from '../utils';
import { useClockGestures } from '../hooks/useClockGestures';

export const AlarmEditModal: React.FC<{
  alarm: Alarm;
  onClose: () => void;
  onSave: (alarm: Alarm) => void;
  onMoreSettings: (hour: number, minute: number) => void;
  onDelete: () => void;
}> = ({ alarm, onClose, onSave, onMoreSettings, onDelete }) => {
  const s = useAppStrings(strings, stringsEn);
  const { bindTap } = useClockGestures();
  const [hour, setHour] = useState(alarm.hour);
  const [minute, setMinute] = useState(alarm.minute);
  const now = getDate();
  const tempAlarm: Alarm = { ...alarm, hour, minute };
  const countdownText = formatCountdownText(getNextTrigger(tempAlarm, now), now.getTime(), s.alarm_rings_in_hours_infix, s.alarm_rings_in_minutes_suffix);

  const handleDone = () => {
    onSave({ ...alarm, hour, minute });
    onClose();
  };

  const handleMoreSettings = () => {
    onMoreSettings(hour, minute);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/25 backdrop-blur-md"
        onClick={onClose}
        aria-hidden
      />
      <div
        className="relative w-full max-w-[340px] bg-app-surface rounded-2xl shadow-xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="px-6 pt-6 pb-2 text-center">
          <h2 className="text-[22px] font-semibold text-black">{s.alarm_modify_title}</h2>
          <p className="text-[15px] text-app-text-muted mt-1">{countdownText}</p>
        </div>
        <div className="flex items-center justify-center gap-6 py-4">
          <div className="flex flex-col items-center">
            <span className="text-[16px] text-app-text-muted mb-2">{s.alarm_hour_label}</span>
            <WheelColumn value={hour} options={Array.from({ length: 24 }, (_, i) => i)} onChange={setHour} medium wrap />
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[16px] text-app-text-muted mb-2">{s.alarm_minute_label}</span>
            <WheelColumn value={minute} options={Array.from({ length: 60 }, (_, i) => i)} onChange={setMinute} medium wrap />
          </div>
        </div>
        <div className="flex gap-3 px-6 pb-6 pt-2">
          <button
            onClick={handleMoreSettings}
            className="flex-1 py-3 rounded-xl bg-gray-100 text-black text-[16px] font-medium"
          >
            {s.alarm_more_settings}
          </button>
          <button
            onClick={handleDone}
            className="flex-1 py-3 rounded-xl bg-app-primary text-white text-[16px] font-medium"
          >
            {s.alarm_done}
          </button>
        </div>
        <div className="px-6 pb-6">
          <button
            className="w-full py-3 rounded-xl bg-red-50 text-red-500 text-[16px] font-medium active:opacity-60"
            {...bindTap<HTMLButtonElement>(
              { kind: 'action', id: 'alarm.edit.delete' },
              { onTrigger: onDelete },
            )}
          >
            {s.toolbar_delete}
          </button>
        </div>
      </div>
    </div>
  );
};
