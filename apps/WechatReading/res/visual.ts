/*
 * Copyright 2026 OPPO
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

import { STANDARD_THEME_ROLES, type VisualAdapter } from '../../../os/visual/types';

export const visualAdapter = {
  appId: 'wechat_reading', label: 'Wechat Reading', version: 2, previewRoute: '/me',
  themeRoles: STANDARD_THEME_ROLES,
  componentRoles: {
    bookshelf_bg: 'surface', bookshelf_header_bg: 'surface', bookshelf_search_input_bg: 'background',
    bookshelf_sort_tab_active: 'accent', bookshelf_item_title: 'text', bookshelf_item_title_selected: 'accent',
    bookshelf_footer_link: 'accent', selection_toolbar_bg: 'surface', selection_toolbar_border: 'border',
    modal_bg: 'surface', modal_title_text: 'text', modal_body_text: 'muted', modal_divider: 'border',
    modal_action_text_primary: 'accent', modal_action_cancel_text: 'text',
    book_detail_bg: 'background', book_detail_nav_text: 'text', book_detail_author_link: 'accent',
    book_detail_card_bg: 'surface', book_detail_action_bar_bg: 'surface',
    tab_bar_bg: 'surface', tab_bar_border: 'border',
    header_bg: 'surface', header_search_bg: 'background',
    settings_bg: 'background', settings_group_bg: 'surface', settings_item_text: 'text',
    settings_item_value_text: 'muted', settings_item_divider: 'border', settings_toggle_on: 'accent',
  },
  layout: { group: 'wechat_reading.primary-nav', kind: 'row4', items: ['reading', 'bookshelf', 'audiobooks', 'me'] },
  coverage: ['Theme tokens and explicitly mapped bookshelf, book detail, modal and navigation colors', 'Consumed settings, privacy and auto-download page backgrounds/text; setting toggle-on colors'],
  protected: ['Reader color presets, content colors, typography and progress', 'Book covers, ranking and shelf order', 'Destructive, disabled and unmapped colors; native user settings'],
} as const satisfies VisualAdapter;
