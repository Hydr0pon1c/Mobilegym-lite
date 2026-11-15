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
import { IcNavBack } from '../../res/icons';
import { dimens } from '../../res/dimens';
import { useWechatReadingGestures } from '../../hooks/useWechatReadingGestures';
import { useWechatReadingStrings } from '../../hooks/useWechatReadingStrings';

const AutoDownloadPage: React.FC = () => {
    const { bindBack } = useWechatReadingGestures();
    const s = useWechatReadingStrings();

    return (
        <div className="flex flex-col h-full bg-(--app-c-settings-bg)">
            {/* Header */}
            <div className="flex items-center px-4 pt-10 pb-3 bg-app-surface sticky top-0 z-10">
                <div className="w-10 flex justify-start" {...bindBack()}>
                    <IcNavBack size={dimens.settings_header_back_size} className="text-(--app-c-settings-item-text)" />
                </div>
                <div className="flex-1 text-center font-bold text-(--app-modal-action-text-size) text-(--app-c-settings-item-text)">{s.auto_download_title}</div>
                <div className="w-10" />
            </div>

            <div className="p-4">
                {/* Upgrade Card */}
                <div className="bg-app-surface rounded-xl p-5 mb-6 flex items-center justify-between shadow-sm">
                    <span className="text-(--app-settings-item-text-size) text-(--app-c-settings-item-text) font-medium">{s.auto_download_member_only}</span>
                    <button className="px-4 py-1.5 bg-(--app-c-settings-auto-download-page-bg-ecdc) text-(--app-c-settings-auto-download-page-text-8b57) rounded-full text-xs font-bold active:opacity-80">
                        {s.auto_download_upgrade}
                    </button>
                </div>

                {/* Info Text */}
                <div className="bg-transparent px-1">
                    <p className="text-xs text-(--app-c-settings-item-value-text) mb-2">{s.auto_download_intro}</p>
                    <ol className="list-decimal list-outside pl-3.5 space-y-2 text-xs text-(--app-c-settings-item-value-text) leading-relaxed">
                        <li>{s.auto_download_rule_1}</li>
                        <li>{s.auto_download_rule_2}</li>
                        <li>{s.auto_download_rule_3}</li>
                        <li>{s.auto_download_rule_4}</li>
                        <li>{s.auto_download_rule_5}</li>
                    </ol>
                </div>
            </div>
        </div>
    );
};

export default AutoDownloadPage;
