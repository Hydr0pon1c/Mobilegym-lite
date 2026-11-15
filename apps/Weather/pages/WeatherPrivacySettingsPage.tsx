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
import { IcNavBack } from '../res/icons';
import { PreferenceCategory } from '../../../system/Settings/components/PreferenceCategory';
import { PreferenceItem } from '../../../system/Settings/components/PreferenceItem';
import { strings } from '../res/strings';
import { stringsEn } from '../res/strings.en';
import { useAppStrings } from '@/os/useAppStrings';
import { useWeatherGestures } from '../hooks/useWeatherGestures';
const lightThemeOverrides: React.CSSProperties = {
    '--app-surface': 'var(--app-c-operation-surface)',
    '--app-text': '#1c1c1e',
    '--app-text-muted': '#8e8e93',
    '--app-primary': 'var(--app-c-settings-toggle-on)',
    '--app-border': '#e5e7eb',
} as React.CSSProperties;

const WeatherPrivacySettingsPage: React.FC = () => {
    const { bindTap, bindBack } = useWeatherGestures();
    const s = useAppStrings(strings, stringsEn);

    return (
        <div
            className="flex flex-col h-full bg-(--app-c-settings-page-bg) dark:bg-black text-black dark:text-white pt-10"
            data-status-bar-foreground="dark"
            style={lightThemeOverrides}
        >
            {/* Header */}
            <div className="flex items-center px-4 py-3 bg-(--app-c-operation-surface) dark:bg-[#1c1c1e] border-b border-gray-200 dark:border-white/10">
                <button
                    {...bindBack<HTMLButtonElement>()}
                    className="p-2 -ml-2 rounded-full hover:bg-black/5 dark:hover:bg-white/5"
                >
                    <IcNavBack size={24} />
                </button>
                <h1 className="ml-2 text-xl font-medium">{s.privacy_title}</h1>
            </div>

            <div className="flex-1 overflow-y-auto pb-8">
                <PreferenceCategory title={s.privacy_title}>
                    <PreferenceItem title={s.privacy_view_policy} />
                    <PreferenceItem title={s.privacy_third_party_data} />
                </PreferenceCategory>

                <PreferenceCategory title={s.privacy_category_permissions}>
                    <PreferenceItem
                        title={s.privacy_permissions_detail}
                        itemProps={bindTap<HTMLDivElement>('permissions.open')}
                    />
                </PreferenceCategory>

                <PreferenceCategory title={s.privacy_category_revoke}>
                    <PreferenceItem title={s.privacy_revoke_consent} />
                </PreferenceCategory>

                <div className="px-4 py-6 text-sm text-gray-400 text-center">
                    {s.privacy_customer_service}
                </div>
            </div>
        </div>
    );
};

export default WeatherPrivacySettingsPage;
