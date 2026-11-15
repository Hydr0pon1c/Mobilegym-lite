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
import { MaskIcon } from './MaskIcon';
import { getSystemSymbolUrl, getSystemDayIconName, IcSymbolMonths, IcSymbolWeeks, IcSymbolYears } from '../res/icons';
import { strings } from '../res/strings';
import { stringsEn } from '../res/strings.en';
import { useAppStrings } from '@/os/useAppStrings';
import { useVisualOrder } from '../../../os/visual/hooks';
export type CalendarViewType = 'year' | 'month' | 'week' | 'day';

interface CalendarBottomNavProps {
    activeView: CalendarViewType;
    onViewChange: (view: CalendarViewType) => void;
    dayNumber: number;
}

const CACHE_V = '2';

export const CalendarBottomNav: React.FC<CalendarBottomNavProps> = ({ activeView, onViewChange, dayNumber }) => {
    const visualOrder = useVisualOrder('calendar');
    const s = useAppStrings(strings, stringsEn);
    const dayIconName = getSystemDayIconName(dayNumber);

    const items: { id: CalendarViewType; label: string; icon: React.ReactNode }[] = [
        {
            id: 'year',
            label: s.tab_year,
            icon: <MaskIcon src={getSystemSymbolUrl(IcSymbolYears, CACHE_V)} size={22} />,
        },
        {
            id: 'month',
            label: s.tab_month,
            icon: <MaskIcon src={getSystemSymbolUrl(IcSymbolMonths, CACHE_V)} size={22} />,
        },
        {
            id: 'week',
            label: s.tab_week,
            icon: <MaskIcon src={getSystemSymbolUrl(IcSymbolWeeks, CACHE_V)} size={22} />,
        },
        {
            id: 'day',
            label: s.tab_day,
            icon: <MaskIcon src={getSystemSymbolUrl(dayIconName, CACHE_V)} size={22} />,
        },
    ];

    return (
        <div data-visual-group="calendar.view-nav" className="flex items-center justify-around py-1.5 border-t border-gray-100 dark:border-gray-800 bg-app-surface dark:bg-black shrink-0 pb-safe">
            {items.map((item) => {
                const active = activeView === item.id;
                return (
                    <button
                        key={item.id}
                        data-visual-item={item.id} style={visualOrder(item.id)}
                        onClick={() => onViewChange(item.id)}
                        className="flex flex-col items-center justify-center gap-0.5 w-14 py-1"
                    >
                        <span className={active ? 'text-black dark:text-white' : 'text-gray-400'}>
                            {item.icon}
                        </span>
                        <span className={`text-[10px] ${active ? 'text-black dark:text-white font-medium' : 'text-gray-400'}`}>
                            {item.label}
                        </span>
                    </button>
                );
            })}
        </div>
    );
};
