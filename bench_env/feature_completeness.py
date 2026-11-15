# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reviewed task-template feature masks; resources and app settings stay intact.

Membership is explicit and independent of sampled parameters or reference paths.
Conditional tasks preserve the union of both branches. Unknown tasks remain full.
"""
from __future__ import annotations

import hashlib
from typing import Any

VERSION = "features_v1"
MIX_MODE = "feature0_mix_v1"
MIX_PERIOD = 8
LAYOUT_MODE = "feature0_layout_v1"
LAYOUT_VERSION = "feature_layout_v1"
SELECTIVE_MODE = "feature0_selective_v1"
PLUS_MODE = "feature0_plus_v1"
# Functional restorations within existing profiles; feature0 itself is frozen.
SELECTIVE_RESTORE = {
    "clock_alarm": ("clock.world",),
    "calendar_dates": ("calendar.settings",),
    "wechat_communication": ("wechat.profile_edit",),
}
LAYOUT_MODULES = frozenset({
    "clock.world", "clock.stopwatch", "clock.timer", "notes.todo", "notes.settings",
    "calendar.date_calculate", "calendar.settings", "wechat.moments", "wechat.profile_edit",
})
STATIC_MODES = ("feature0", "feature1")
MODES = (*STATIC_MODES, MIX_MODE, LAYOUT_MODE, SELECTIVE_MODE, PLUS_MODE)

# A profile is the set of safely removable modules, NOT a required-route whitelist.
PROFILES = {
    "clock_alarm": ('clock.world', 'clock.stopwatch', 'clock.timer'),
    "clock_world": ('clock.stopwatch', 'clock.timer'),
    "notes_note": ('notes.todo',),
    "notes_todo": ('notes.settings',),
    "calendar_fixed": ('calendar.date_calculate',),
    "calendar_dates": ('calendar.settings',),
    "wechat_communication": ('wechat.moments', 'wechat.profile_edit'),
    "wechat_personal": ('wechat.moments',),
    "wechat_social": ('wechat.profile_edit',),
}

# Audited against task source; preserve launch pages and ordinary navigation.
PROFILE_TASKS = {
    "clock_alarm": frozenset("""
clock.CheckAlarmNote
clock.DeleteAlarm
clock.SetAlarmRepeat
clock.SetupMorningAlarms
clock.ToggleAlarm
crossapp_work.WeatherConditionalCancelMeeting
""".split()),
    "clock_world": frozenset("""
clock.AddCityAndCompareTimeDiff
clock.AddWorldCity
clock.CheckCityTime
clock.CompareCityTimeDiff
clock.LatestTimezoneCity
clock.RemoveWorldCity
""".split()),
    "notes_note": frozenset("""
account.Railway12306LoginWithAccount
crossapp_commerce.AlipayMonthlyToNotesAndWechat
crossapp_commerce.EbayLowestPriceToNotes
crossapp_content.SpotifySaveCurrentSongToNotes
crossapp_life.RealisticTrip001
crossapp_life.TripMemoAndNotify
notes.CreateFolderAndMoveNote
notes.CreateNewNote
notes.CreateNoteWithReminder
notes.PinNote
notes.ReadNoteContent
notes.RestoreFromTrash
notes.SearchNoteTitle
""".split()),
    "notes_todo": frozenset("""
notes.AddNewTodo
notes.DeleteAllCompletedTodos
notes.DeleteTodo
notes.ReadTodoText
""".split()),
    "calendar_fixed": frozenset("""
calendar.ChangeDefaultReminder
calendar.ConfigAllReminders
calendar.CreateBirthdayEvent
calendar.CreateEvent
calendar.CreateEventWithReminder
calendar.DeleteEvent
calendar.QueryFirstEventOnDate
calendar.SearchDeleteAll
calendar.ToggleShowWeekNumber
""".split()),
    "calendar_dates": frozenset("""
calendar.CalculateDateInterval
calendar.DateCalcForward
calendar.MakeupDayReminder
calendar.QueryHolidayLength
crossapp_life.CalendarEventToWechat
crossapp_life.CalendarFreeWeatherInvite
crossapp_life.RestaurantRatingInviteCalendar
crossapp_life.WeatherFirstNonRainyToCalendarAndSms
crossapp_life.WeekendTripFullPlan
crossapp_work.SmsAndCalendarOnDate
""".split()),
    "wechat_communication": frozenset("""
crossapp_commerce.AlipayBalanceToWechat
crossapp_commerce.AlipayMonthlySpendToWechat
crossapp_commerce.AlipayMonthlyToNotesAndWechat
crossapp_commerce.AlipayShareBillDetail
crossapp_content.BilibiliRankAuthorLastNovToWechat
crossapp_content.BilibiliRankingToWechat
crossapp_content.RedbookAuthorFollowersToWechat
crossapp_content.RedbookDmThenWechatReport
crossapp_content.SpotifyNowPlayingToWechat
crossapp_content.WechatReadingBestBookToWechat
crossapp_content.WechatReadingShareBookList
crossapp_life.CalendarEventToWechat
crossapp_life.CalendarFreeWeatherInvite
crossapp_life.RailwayTomorrowMomBookingToWechat
crossapp_life.RailwayWeatherToWechat
crossapp_life.RealisticTrip001
crossapp_life.RestaurantRatingInviteCalendar
crossapp_life.TripMemoAndNotify
crossapp_life.WeatherRainBranchNotify
crossapp_life.WeekendTripFullPlan
crossapp_work.MeetingLongestInfoToWechat
crossapp_work.MeetingRouteEtaToWechat
wechat.OpenRadarAddFriend
wechat.ReadContactRegion
""".split()),
    "wechat_personal": frozenset("""
wechat.DeauthorizeApp
wechat.SetPatText
wechat.SetSignature
wechat.ToggleFriendConfirmation
wechat.ToggleWechatSports
""".split()),
    "wechat_social": frozenset("""
crossapp_content.BilibiliTripleLikeThenMoments
wechat.BlacklistContact
wechat.PostMomentsText
wechat.PostMomentsTextWithCity
wechat.SetFriendChatOnly
wechat.SetMomentsVisibleRange
wechat.ToggleMobileAutoPlayMomentsVideo
wechat.ToggleStrangerViewMoments
""".split()),
}
TASK_IDS = frozenset().union(*PROFILE_TASKS.values())

# Reviewed only for the maintained training templates; future/unknown tasks stay full.
PLUS_PROFILES = {
    "bilibili_commerce": ("bilibili.vip", "bilibili.shop"),
    "spotify_premium": ("spotify.premium",),
    "alipay_video": ("alipay.video",),
}
PLUS_PROFILE_TASKS = {
    'bilibili_commerce': frozenset("""
bilibili.CoinVideoTask
bilibili.FavVideoAndCountTask
bilibili.FollowRecommendationTask
bilibili.SanlianTask
bilibili.SearchUserFollowerCountTask
bilibili.SetBirthdayTask
bilibili.SubscribeTask
bilibili.UnfollowAndClearHistoryTask
bilibili.ViewFavoritesFolderCountTask
crossapp_content.BilibiliRankAuthorLastNovToWechat
crossapp_content.BilibiliRankingToWechat
crossapp_content.BilibiliTripleLikeThenMoments
""".split()),
    'spotify_premium': frozenset("""
crossapp_content.SpotifyNowPlayingToWechat
crossapp_content.SpotifySaveCurrentSongToNotes
spotify.AddToQueueAndPlay
spotify.BuildPlaylistFromTwoArtists
spotify.CollectLikedRecentAndPlay
spotify.CreateNewPlaylist
spotify.DiscoverSaveAndReport
spotify.FollowAndPlayArtist
spotify.QueueAndLikeSong
spotify.QueueTopArtistSongs
spotify.SearchAlbumInfo
spotify.SearchPlayAndReport
spotify.TogglePrivacy
""".split()),
    'alipay_video': frozenset("""
alipay.AnalyzeSpending
alipay.CheckLatestMessageContent
alipay.CountLargeTransferIncomes
alipay.DisableAllNotifications
alipay.EnableDarkMode
alipay.MonthlyIncomeByCounterparty
alipay.SendMessageToContact
alipay.SetFontSizeLevel
alipay.SetPayOrderCcbYuebaoBalance
alipay.ShowReceiveQRCode
crossapp_commerce.AlipayBalanceToWechat
crossapp_commerce.AlipayMonthlySpendToWechat
crossapp_commerce.AlipayMonthlyToNotesAndWechat
crossapp_commerce.AlipayShareBillDetail
""".split()),
}
PLUS_TASK_IDS = TASK_IDS | frozenset().union(*PLUS_PROFILE_TASKS.values())



def feature_plan(task: Any, mode: str, *, rollout_index: int = 0) -> dict:
    from bench_env.completeness import base_task_id
    if mode not in MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    tid = base_task_id(task)
    profiles = sorted(name for name, ids in PROFILE_TASKS.items() if tid in ids)
    candidates = sorted({feature for name in profiles for feature in PROFILES[name]})
    if mode == PLUS_MODE:
        extra = sorted(name for name, ids in PLUS_PROFILE_TASKS.items() if tid in ids)
        candidates = sorted(set(candidates) | {feature for name in extra for feature in PLUS_PROFILES[name]})
        profiles = sorted(profiles + extra)
    # feature0 removes all reviewed candidates. Only feature1 ranks them.
    kept = []
    mixture = None
    if mode == "feature1":
        # Stable nested masks; never use parameters, answers or global RNG.
        ordered = sorted(candidates, key=lambda feature:
            hashlib.sha256(f"{VERSION}:{tid}:{feature}".encode()).digest())
        kept = sorted(ordered[:(len(ordered) + 1) // 2])
    elif mode in (SELECTIVE_MODE, PLUS_MODE):
        restored = {feature for name in profiles for feature in SELECTIVE_RESTORE.get(name, ())}
        kept = sorted(set(candidates).intersection(restored))
    elif mode == MIX_MODE:
        if type(rollout_index) is not int or rollout_index < 0:
            raise ValueError("Feature mixture requires a nonnegative integer rollout index")
        # Exactly one full-context rollout per eight for every supported template.
        # Task identity chooses its slot; task parameters, answers and RNG do not.
        digest = hashlib.sha256(f"{MIX_MODE}:{tid}".encode()).digest()
        full_slot = int.from_bytes(digest[:8], "big") % MIX_PERIOD
        slot = rollout_index % MIX_PERIOD
        full = slot == full_slot
        kept = candidates if full else []
        mixture = {"period": MIX_PERIOD, "full_slot": full_slot,
                   "rollout_index": rollout_index, "slot": slot,
                   "applied_mode": "full" if full or not candidates else "feature0"}
    disabled = sorted(set(candidates) - set(kept))
    return {
        "version": VERSION, "mode": mode, "task_id": tid,
        "supported": bool(profiles), "effective": bool(disabled), "profiles": profiles,
        "candidates": candidates, "kept": kept, "disabled": disabled,
        "remaining": len(kept), "total": len(candidates),
        "remaining_fraction": len(kept) / len(candidates) if candidates else None,
        **({"mixture": mixture} if mixture is not None else {}),
        **({"layout": {
            "version": LAYOUT_VERSION, "mode": "preserve",
            "effective": bool(LAYOUT_MODULES.intersection(disabled)),
            "modules": sorted(LAYOUT_MODULES.intersection(disabled)),
        }} if mode == LAYOUT_MODE else {}),
    }


async def apply_features(env: Any, task: Any, obs: Any, mode: str, *, rollout_index: int = 0) -> Any:
    if not getattr(env, "supports_state_injection", True) or getattr(task, "_instruction_override", None) is not None:
        raise ValueError("Feature masks require an unmodified simulator task")
    record = feature_plan(task, mode, rollout_index=rollout_index)
    # Fail on an old frozen frontend instead of silently training the full condition.
    actual = await env.page.evaluate("""({version, disabled, layout}) => {
        const api = window.__SIM__;
        if (api?.getFeatureAvailability?.().version !== version) {
            throw new Error('Feature masks require a matching features_v1 frontend build');
        }
        if (layout && api.getFeatureLayout?.().version !== layout.version) {
            throw new Error('Feature layout requires a matching feature_layout_v1 frontend build');
        }
        const mask = api.setDisabledFeatures(disabled,
            layout ? {layout: layout.mode} : undefined);
        return layout ? {...mask, layout: api.getFeatureLayout()} : mask;
    }""", record)
    expected = {"version": VERSION, "disabled": record["disabled"]}
    if "layout" in record:
        expected["layout"] = {"version": LAYOUT_VERSION, "mode": "preserve"}
    if actual != expected:
        raise ValueError("Feature mask readback mismatch")
    task._completeness_record = record
    return await env.get_observation()
