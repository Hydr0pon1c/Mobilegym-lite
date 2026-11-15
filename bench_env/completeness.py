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

"""Opt-in, post-sampling resource projections and MobileGym Lite setup.

Rules preserve complete rows and relative order. They do not constrain actions or
rewards. Keep this bounded: add a task to TASK_IDS only after auditing its inputs,
query domain and conditional branches; do not infer support from a gold path.
"""
from __future__ import annotations

from copy import deepcopy
import datetime
import hashlib
import re
from typing import Any
from bench_env.feature_completeness import MODES as FEATURE_MODES, apply_features

VERSION = "resources_v1"
MODES = {"resources_full": None, "resources_2": 2, "resources_1": 1, "resources_0": 0}
# Version the composition independently; existing ablation modes stay unchanged.
LITE_VERSION = "lite_v1"
LITE_RESOURCE_MODE = "resources_2"
LITE_FEATURE_MODE = "feature0"
COMPLETENESS_MODES = (*MODES, *FEATURE_MODES, LITE_VERSION)
APPS = frozenset({"wechat", "notes", "clock", "calendar", "alipay", "tencent_meeting"})

# Explicit v3 candidate snapshot; generated once from the audited app inventory.
TASK_IDS = frozenset("""
account.Railway12306LoginWithAccount
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
calendar.CalculateDateInterval
calendar.ChangeDefaultReminder
calendar.ConfigAllReminders
calendar.CreateBirthdayEvent
calendar.CreateEvent
calendar.CreateEventWithReminder
calendar.DateCalcForward
calendar.DeleteEvent
calendar.MakeupDayReminder
calendar.QueryFirstEventOnDate
calendar.QueryHolidayLength
calendar.SearchDeleteAll
calendar.ToggleShowWeekNumber
clock.AddCityAndCompareTimeDiff
clock.AddWorldCity
clock.CheckAlarmNote
clock.CheckCityTime
clock.CompareCityTimeDiff
clock.DeleteAlarm
clock.LatestTimezoneCity
clock.RemoveWorldCity
clock.SetAlarmRepeat
clock.SetupMorningAlarms
clock.ToggleAlarm
crossapp_commerce.AlipayBalanceToWechat
crossapp_commerce.AlipayMonthlySpendToWechat
crossapp_commerce.AlipayMonthlyToNotesAndWechat
crossapp_commerce.AlipayShareBillDetail
crossapp_commerce.EbayLowestPriceToNotes
crossapp_content.BilibiliRankAuthorLastNovToWechat
crossapp_content.BilibiliRankingToWechat
crossapp_content.BilibiliTripleLikeThenMoments
crossapp_content.RedbookAuthorFollowersToWechat
crossapp_content.RedbookDmThenWechatReport
crossapp_content.SpotifyNowPlayingToWechat
crossapp_content.SpotifySaveCurrentSongToNotes
crossapp_content.WechatReadingBestBookToWechat
crossapp_content.WechatReadingShareBookList
crossapp_life.CalendarEventToWechat
crossapp_life.CalendarFreeWeatherInvite
crossapp_life.RailwayTomorrowMomBookingToWechat
crossapp_life.RailwayWeatherToWechat
crossapp_life.RealisticTrip001
crossapp_life.RestaurantRatingInviteCalendar
crossapp_life.TripMemoAndNotify
crossapp_life.WeatherFirstNonRainyToCalendarAndSms
crossapp_life.WeatherRainBranchNotify
crossapp_life.WeekendTripFullPlan
crossapp_work.MeetingJoinAndNotifySms
crossapp_work.MeetingLongestInfoToWechat
crossapp_work.MeetingRouteEtaToWechat
crossapp_work.SmsAndCalendarOnDate
crossapp_work.WeatherConditionalCancelMeeting
notes.AddNewTodo
notes.CreateFolderAndMoveNote
notes.CreateNewNote
notes.CreateNoteWithReminder
notes.DeleteAllCompletedTodos
notes.DeleteTodo
notes.PinNote
notes.ReadNoteContent
notes.ReadTodoText
notes.RestoreFromTrash
notes.SearchNoteTitle
tencent_meeting.CheckContactCount
tencent_meeting.CheckPersonalRoomId
tencent_meeting.FindMeetingWithMostParticipants
tencent_meeting.JoinMeetingAndRename
tencent_meeting.StartFastMeeting
tencent_meeting.ToggleNotification
wechat.BlacklistContact
wechat.DeauthorizeApp
wechat.OpenRadarAddFriend
wechat.PostMomentsText
wechat.PostMomentsTextWithCity
wechat.ReadContactRegion
wechat.SetFriendChatOnly
wechat.SetMomentsVisibleRange
wechat.SetPatText
wechat.SetSignature
wechat.ToggleFriendConfirmation
wechat.ToggleMobileAutoPlayMomentsVideo
wechat.ToggleStrangerViewMoments
wechat.ToggleWechatSports
""".split())


def validate_completeness(mode: str | None, device: str = "sim") -> None:
    if mode is not None and mode not in COMPLETENESS_MODES:
        raise ValueError(f"Unknown completeness mode {mode!r}; choose {sorted(COMPLETENESS_MODES)}")
    if mode is not None and device != "sim":
        raise ValueError("Local completeness requires the simulator")


def base_task_id(task: Any) -> str:
    return re.sub(r"_i\d+$", "", task.id)


class _Projection:
    def __init__(self, task_id: str, state: dict, extra: int | None):
        self.task_id, self.state, self.extra = task_id, state, extra
        self.patch: dict[str, dict] = {}
        self.collections: list[dict] = []

    def select(self, app: str, key: str, protected: list[dict], reason: str,
               *, id_key: str = "id", background: bool = False) -> list[dict]:
        rows = self.state["apps"][app][key]
        ids = [r[id_key] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {app}.{key}")
        protected_ids = {r[id_key] for r in protected}
        if not protected_ids <= set(ids) or any(r not in rows for r in protected):
            raise ValueError(f"Protected rows do not resolve in {app}.{key}")
        # Separate from task sampling; stable across processes and list ordering.
        # The same ordering gives nested full / 2 / 1 / 0 conditions.
        candidates = sorted((i for i in ids if i not in protected_ids), key=lambda i:
            hashlib.sha256(f"{VERSION}:{self.task_id}:{app}:{key}:{i}".encode()).digest())
        keep = protected_ids | set(candidates if self.extra is None else candidates[:self.extra])
        selected = [r for r in rows if r[id_key] in keep]
        self.put(app, key, selected, reason, id_key=id_key,
                 protected_ids=protected_ids, background=background)
        return selected

    def put(self, app: str, key: str, rows: list[dict], reason: str, *, id_key: str = "id",
            protected_ids: set | None = None, background: bool = False) -> None:
        original = self.state["apps"][app][key]
        if self.extra is None:
            rows = original
        self.patch.setdefault(app, {})[key] = deepcopy(rows)
        self.collections.append({
            "app": app, "collection": key, "before": len(original), "after": len(rows),
            "protected_ids": sorted(protected_ids or (), key=str),
            "kept_ids": [r[id_key] for r in rows], "reason": reason,
            "background_only": background,
        })


def _require(rows: list[dict], label: str) -> list[dict]:
    if not rows:
        raise ValueError(f"Resource target did not resolve: {label}")
    return rows


def _wechat(p: _Projection, task: Any) -> None:
    s = p.state["apps"]["wechat"]
    name = ("母亲" if p.task_id == "crossapp_life.RailwayTomorrowMomBookingToWechat"
            else task.params.get("contact"))
    if name is None and p.task_id in {
        "wechat.PostMomentsText", "wechat.PostMomentsTextWithCity",
        "crossapp_content.BilibiliTripleLikeThenMoments",
    }:
        p.select("wechat", "moments", [], "Create a new post; existing posts are background")
        return
    targets = (_require([c for c in s["contacts"] if c["name"] == name], str(name))
               if name is not None else [])
    contacts = p.select("wechat", "contacts", targets, "Target contact, or background for profile/settings",
                        id_key="wxid", background=name is None)
    kept = {c["wxid"] for c in contacts} | {s["user"]["wxid"]}
    p.put("wechat", "chats", [c for c in s["chats"] if c["id"] in kept],
          "Keep existing conversations of retained contacts and self", background=name is None)
    p.put("wechat", "moments", [m for m in s["moments"] if m.get("wxid") in kept],
          "Keep posts of retained contacts and self", background=True)


def _notes(p: _Projection, task: Any) -> None:
    s, params = p.state["apps"]["notes"], task.params
    if p.task_id in {"notes.ReadTodoText", "notes.DeleteAllCompletedTodos"}:
        p.select("notes", "todos", s["todos"], "Preserve full quantified/preservation domain")
    elif p.task_id in {"notes.DeleteTodo", "notes.AddNewTodo"}:
        text = params.get("todo_text", params.get("text"))
        targets = [r for r in s["todos"] if r["text"] == text]
        if p.task_id == "notes.DeleteTodo":
            _require(targets, str(text))
        p.select("notes", "todos", targets, "Target todo; preserve existing matches on creation")
    else:
        title = params.get("note_title", params.get("noteTitle", params.get("title")))
        targets = [r for r in s["notes"] if r["title"] == title]
        if p.task_id == "notes.SearchNoteTitle":
            query = params["keyword"].lower()
            targets = [r for r in s["notes"]
                       if query in (r["title"] + "\n" + r.get("content", "")).lower()]
        if "note_title" in params or "noteTitle" in params:
            _require(targets, str(title))
        p.select("notes", "notes", targets, "Target note or complete matching search set; retain folders")


def _clock(p: _Projection, task: Any) -> None:
    alarms = p.state["apps"]["clock"]["alarms"]
    target = task.params.get("alarm_id")
    protected = _require([r for r in alarms if r["id"] == target], str(target)) if target else []
    world = p.task_id in {"clock.AddCityAndCompareTimeDiff", "clock.AddWorldCity", "clock.CheckCityTime",
                         "clock.CompareCityTimeDiff", "clock.LatestTimezoneCity", "clock.RemoveWorldCity"}
    p.select("clock", "alarms", protected, "Target alarm; city catalogs and selected cities stay complete",
             background=world)


def _calendar(p: _Projection, task: Any) -> None:
    from bench_env.task.calendar.app import Calendar
    s = p.state["apps"]["calendar"]
    cal = Calendar(s)
    if p.task_id == "calendar.DeleteEvent":
        protected = _require([r for r in s["events"] if r["title"] == task.params["title"]], "event title")
    elif p.task_id == "calendar.SearchDeleteAll":
        protected = _require(cal.find_events_by_keyword(task.params["keyword"]), "event search")
    elif p.task_id == "calendar.QueryFirstEventOnDate":
        protected = _require(cal.find_events_on_date(Calendar.parse_ymd(task.params["date"])), "events on date")
    else:
        # Usually empty; preserve all for cross-app availability/conditional tasks.
        protected = s["events"]
    p.select("calendar", "events", protected, "Preserve target/search/date domain; other families keep all events")


def _alipay(p: _Projection, task: Any) -> None:
    s, tid = p.state["apps"]["alipay"], p.task_id
    if tid == "alipay.SendMessageToContact":
        targets = _require([c for c in s["contacts"] if c["name"] == task.params["contact"]], "Alipay contact")
        contacts = p.select("alipay", "contacts", targets, "Target contact")
        ids = {c["id"] for c in contacts}
        p.put("alipay", "conversations", [c for c in s["conversations"]
              if c.get("kind") != "person" or c.get("contactId") in ids], "Retained contacts plus service conversations")
        # chatHistory is an ID-keyed map: deep setState cannot delete its keys.
        # Keep its contents; this intervention concerns visible contact/conversation lists.
        return
    if tid == "alipay.CheckLatestMessageContent":
        target = _require([c for c in s["conversations"] if c["name"] == task.params["name"]], "Alipay conversation")
        p.select("alipay", "conversations", target, "Whole target conversation; messages and contacts stay complete")
        return
    rows = s["transferRecords"]
    background = False
    if tid == "alipay.AnalyzeSpending":
        protected = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:5]
    elif tid == "alipay.CountLargeTransferIncomes":
        protected = rows  # Keep all counting candidates, not only positive matches.
    elif tid in {"alipay.MonthlyIncomeByCounterparty", "crossapp_commerce.AlipayMonthlySpendToWechat",
                 "crossapp_commerce.AlipayMonthlyToNotesAndWechat"}:
        month = task.params.get("month") or datetime.datetime.fromtimestamp(
            p.state["os"]["time"]["timestamp"] / 1000).strftime("%Y-%m")
        protected = [r for r in rows if datetime.datetime.fromtimestamp(r["timestamp"] / 1000).strftime("%Y-%m") == month]
    elif tid == "crossapp_commerce.AlipayShareBillDetail":
        ordered = sorted(rows, key=lambda r: r["timestamp"], reverse=True)
        first = next((i for i, r in enumerate(ordered) if float(r["delta"]) < 0), None)
        if first is None:
            raise ValueError("No latest expense")
        protected = ordered[:first + 1]  # Keep the ranking prefix, including income rows.
    else:
        protected, background = [], True  # Profile/settings, balance and QR tasks.
    p.select("alipay", "transferRecords", protected, "Full query domain, latest prefix, or background bills",
             background=background)


def _meeting(p: _Projection, task: Any) -> None:
    from bench_env.task.tencent_meeting.app import TencentMeeting
    s, tid = p.state["apps"]["tencent_meeting"], p.task_id
    tm = TencentMeeting(s)
    if tid == "tencent_meeting.CheckContactCount":
        p.select("tencent_meeting", "contacts", s["contacts"], "Preserve full counted contact set")
        return
    if tid == "tencent_meeting.FindMeetingWithMostParticipants":
        protected = _require([m for m in s["history"] if str(m.get("hostId")) == tm.user_id], "hosted meetings")
        background = False
    elif tid == "crossapp_work.MeetingLongestInfoToWechat":
        protected = _require(tm.history_meetings_on_date(task.params["date"]), "meetings on date")
        background = False
    else:
        protected, background = [], True
    p.select("tencent_meeting", "history", protected, "Full comparison set, or background history",
             background=background)
    # Ongoing/scheduled meetings and participants stay complete for both branches.


_RULES = {"wechat": _wechat, "notes": _notes, "clock": _clock,
          "calendar": _calendar, "alipay": _alipay, "tencent_meeting": _meeting}


def project_resources(task: Any, state: dict, mode: str) -> tuple[dict, dict]:
    validate_completeness(mode)
    if mode not in MODES:
        raise ValueError(f"Not a resource mode: {mode}")
    tid = base_task_id(task)
    supported = tid in TASK_IDS
    p = _Projection(tid, state, MODES[mode])
    if supported:
        for app in sorted(set(task.apps) & APPS):
            _RULES[app](p, task)
    changed = [c for c in p.collections if c["after"] < c["before"]]
    return {"apps": p.patch}, {
        "version": VERSION, "mode": mode, "task_id": tid, "supported": supported,
        "effective": bool(changed), "background_only": bool(changed) and all(c["background_only"] for c in changed),
        "collections": p.collections,
    }


def query_answer(task: Any, obs: Any) -> tuple[bool, Any]:
    """Available query oracle only; compound operation goals need GUI checks."""
    from bench_env.task.judge import JudgeInput
    if base_task_id(task) == "wechat.ReadContactRegion":
        contacts = obs.state["apps"]["wechat"]["contacts"]
        return True, next(c["region"] for c in contacts if c["name"] == task.params["contact"])
    getter = getattr(task, "get_answer", None)
    if getattr(task, "answer_fields", None) and callable(getter):
        try:
            return True, getter(JudgeInput(init_obs=obs, last_obs=obs))
        except NotImplementedError:
            pass
    return False, None


async def apply_resources(env: Any, task: Any, obs: Any, mode: str | None) -> Any:
    task._completeness_record = None
    validate_completeness(mode)
    if mode is None:
        return obs
    if not getattr(env, "supports_state_injection", True) or getattr(task, "_instruction_override", None) is not None:
        raise ValueError("Resource projections require an unmodified simulator task")
    patch, record = project_resources(task, obs.state, mode)
    task._completeness_record = record
    if not record["effective"]:
        return obs
    answer = query_answer(task, obs)
    await env.set_state(patch, deep=True, reload=False)
    reduced = await env.get_observation()
    for app, fields in patch["apps"].items():
        for key, expected in fields.items():
            if reduced.state["apps"][app].get(key) != expected:
                raise ValueError(f"Resource injection readback mismatch: {app}.{key}")
    if query_answer(task, reduced) != answer:
        raise ValueError(f"Resource projection changed the query answer: {base_task_id(task)}")
    record["query_answer_checked"] = answer[0]
    return reduced


async def apply_completeness(env: Any, task: Any, obs: Any, mode: str | None, *, rollout_index: int = 0) -> Any:
    validate_completeness(mode)
    if mode in FEATURE_MODES:
        return await apply_features(env, task, obs, mode, rollout_index=rollout_index)
    # Also clear on full/unsupported episodes when a caller reuses a page without
    # a hard reset. Older resource-only frozen builds need no feature API.
    page = getattr(env, "page", None)
    if page is not None:
        changed = await page.evaluate("""() => {
            const changed = !!window.__SIM__?.getFeatureAvailability?.().disabled.length;
            window.__SIM__?.setDisabledFeatures?.([]);
            return changed;
        }""")
        if changed:
            obs = await env.get_observation()
    reduced = await apply_resources(env, task, obs, LITE_RESOURCE_MODE if mode == LITE_VERSION else mode)
    if mode != LITE_VERSION:
        return reduced

    # Install Lite's feature mask after resource readback, before the first
    # policy observation. Keep both audit records; neither changes the verifier.
    resources = task._completeness_record
    final = await apply_features(env, task, reduced, LITE_FEATURE_MODE)
    features = task._completeness_record
    if final.state["apps"] != reduced.state["apps"]:
        raise ValueError("Lite feature mask changed app data after resource projection")
    task._completeness_record = {
        "version": LITE_VERSION, "mode": mode, "task_id": base_task_id(task),
        "supported": resources["supported"] or features["supported"],
        "effective": resources["effective"] or features["effective"],
        "background_only": resources["background_only"] and not features["effective"],
        "resources": resources, "features": features,
    }
    return final
