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

import React, { useEffect } from 'react';
import { Feature, FeatureSlot, type FeatureId } from '@/os/featureAvailability';

export interface CalendarActionSheetItem {
  id: string;
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  onClick: () => void;
  feature?: FeatureId;
}

export const CalendarActionSheet: React.FC<{
  open: boolean;
  title?: string;
  items: CalendarActionSheetItem[];
  onClose: () => void;
}> = ({ open, title, items, onClose }) => {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <button
        className="absolute inset-0 bg-black/40"
        aria-label="关闭"
        onClick={onClose}
      />
      <div className="absolute left-0 right-0 bottom-0 pb-safe">
        <div className="mx-3 mb-3 rounded-3xl overflow-hidden bg-app-surface dark:bg-[#1c1c1e] shadow-xl">
          {title && (
            <div className="px-5 pt-4 pb-2 text-[13px] text-app-text-muted dark:text-gray-400">
              {title}
            </div>
          )}
          <div className="divide-y divide-gray-100 dark:divide-white/10">
            {items.map((it) => {
              const content = <>
                {it.icon && <div className="shrink-0 text-gray-700 dark:text-gray-200">{it.icon}</div>}
                <div className="min-w-0 flex-1">
                  <div className="text-[16px] text-app-text dark:text-gray-100 truncate">{it.title}</div>
                  {it.subtitle && <div className="text-[12px] text-gray-400 mt-0.5 truncate">{it.subtitle}</div>}
                </div>
              </>;
              const button = <button
                key={it.id}
                className="w-full flex items-center gap-3 px-5 py-4 text-left active:bg-black/5 dark:active:bg-white/5"
                onClick={() => {
                  it.onClick();
                  onClose();
                }}
              >
                {content}
              </button>;
              return it.feature ? <Feature key={it.id} id={it.feature} placeholder={
                <FeatureSlot id={it.feature} className="w-full flex items-center gap-3 px-5 py-4 text-left">
                  {content}
                </FeatureSlot>
              }>{button}</Feature> : button;
            })}
          </div>
        </div>

        <div className="mx-3 mb-3 rounded-3xl overflow-hidden bg-app-surface dark:bg-[#1c1c1e] shadow-xl">
          <button
            className="w-full px-5 py-4 text-[16px] text-app-text dark:text-gray-100 active:bg-black/5 dark:active:bg-white/5"
            onClick={onClose}
          >
            取消
          </button>
        </div>
      </div>
    </div>
  );
};
