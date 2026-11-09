from datetime import datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="달러 세븐스플릿 V4 백테스트", page_icon="💵", layout="wide")
KST=ZoneInfo("Asia/Seoul")
st.title("💵 달러 세븐스플릿 V4 · 실제 7차 매수·매도 백테스트")
st.caption("USD/KRW 5분봉의 고가·저가를 이용해 1차 진입, -간격 추가매수, +목표수익 차수별 매도를 순서대로 모의체결합니다.")
st.warning("백테스트는 교육·연구용입니다. Yahoo Finance 공개 호가와 실제 은행 환전가는 다르며, 과거 성과는 미래 수익을 보장하지 않습니다.")

@st.cache_data(ttl=900, show_spinner=False)
def load_fx(period, interval):
    d=yf.download("KRW=X",period=period,interval=interval,progress=False,auto_adjust=False,prepost=True,threads=False)
    if d is None or d.empty: raise RuntimeError("USD/KRW 데이터를 가져오지 못했습니다.")
    if isinstance(d.columns,pd.MultiIndex): d.columns=[x[0] for x in d.columns]
    ix=pd.DatetimeIndex(d.index)
    if ix.tz is None: ix=ix.tz_localize("UTC")
    d.index=ix.tz_convert(KST)
    return d.sort_index().dropna(subset=["Open","High","Low","Close"])

def daily_signal_table(bars, entry_t, ma_days, discount, low_lookback, low_band, reversal):
    rows=[]
    for day,g in bars.groupby(bars.index.date):
        target=pd.Timestamp(datetime.combine(day,entry_t),tz=KST)
        pos=g.index.get_indexer([target],method="nearest")[0]
        if pos<0 or abs(g.index[pos]-target)>pd.Timedelta(minutes=20): continue
        rows.append({"date":pd.Timestamp(day),"timestamp":g.index[pos],"close":float(g.iloc[pos]["Close"])})
    x=pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    x["ma"]=x["close"].rolling(ma_days,min_periods=ma_days).mean()
    x["low_n"]=x["close"].rolling(low_lookback,min_periods=low_lookback).min()
    x["ma_ok"]=x["close"]<=x["ma"]*(1-discount/100)
    x["low_ok"]=x["close"]<=x["low_n"]*(1+low_band/100)
    x["reversal_ok"]=x["close"]>x["close"].shift(1) if reversal else True
    x["signal"]=x["ma_ok"] & x["low_ok"] & x["reversal_ok"]
    return x

def simulate(bars, signals, tranche_amt, levels, gap_pct, take_pct, cost_pct):
    sig={r.timestamp:r for _,r in signals[signals.signal].iterrows()}
    active=None; cycles=[]; trades=[]; cashflow=0.0; max_cap=0.0; capital=0.0; cid=0
    for ts,r in bars.iterrows():
        # 기존 보유 차수 매도부터 처리. 같은 봉에서 새로 산 차수는 즉시 팔지 않는 보수적 규칙.
        if active:
            sold=[]
            for tr in active["tranches"]:
                if tr["open"] and float(r.High)>=tr["target"]:
                    proceeds=tranche_amt*(1+take_pct/100)*(1-cost_pct/100)
                    pnl=proceeds-tranche_amt
                    capital-=tranche_amt; cashflow+=pnl
                    tr.update(open=False,sell_time=ts,sell_price=tr["target"],pnl=pnl)
                    trades.append({"사이클":active["id"],"차수":tr["level"],"구분":"매도","시각":ts,"환율":tr["target"],"금액":tranche_amt,"실현손익":pnl})
                    sold.append(tr["level"])
            if active["opened"]>0 and not any(t["open"] for t in active["tranches"]):
                active["end"]=ts; active["pnl"]=sum(t.get("pnl",0) for t in active["tranches"]); active["max_level"]=active["opened"]
                cycles.append(active); active=None
        if active:
            # 다음 차수는 정해진 순서로 한 봉당 최대 한 차수만 체결
            nxt=active["opened"]+1
            if nxt<=levels:
                level_price=active["entry"]*((1-gap_pct/100)**(nxt-1))
                if float(r.Low)<=level_price:
                    target=level_price*(1+take_pct/100)
                    tr={"level":nxt,"buy_time":ts,"buy_price":level_price,"target":target,"open":True}
                    active["tranches"].append(tr);active["opened"]+=1;capital+=tranche_amt;max_cap=max(max_cap,capital)
                    trades.append({"사이클":active["id"],"차수":nxt,"구분":"매수","시각":ts,"환율":level_price,"금액":tranche_amt,"실현손익":0.0})
        if active is None and ts in sig:
            cid+=1; entry=float(sig[ts].close); target=entry*(1+take_pct/100)
            tr={"level":1,"buy_time":ts,"buy_price":entry,"target":target,"open":True}
            active={"id":cid,"start":ts,"entry":entry,"opened":1,"tranches":[tr]}
            capital+=tranche_amt;max_cap=max(max_cap,capital)
            trades.append({"사이클":cid,"차수":1,"구분":"매수","시각":ts,"환율":entry,"금액":tranche_amt,"실현손익":0.0})
    # 미청산 평가손익
    last=float(bars.Close.iloc[-1]); unreal=0.0
    if active:
        for tr in active["tranches"]:
            if tr["open"]: unreal += tranche_amt*(last/tr["buy_price"]-1-cost_pct/100)
        active["end"]=pd.NaT;active["pnl"]=sum(t.get("pnl",0) for t in active["tranches"]);active["max_level"]=active["opened"]
        cycles.append(active)
    return pd.DataFrame(trades),cycles,cashflow,unreal,max_cap,last

with st.sidebar:
    st.header("⚙️ 매매 규칙")
    period=st.selectbox("백테스트 기간",["60d","30d"],0)
    interval=st.selectbox("봉 간격",["5m","15m"],0)
    entry_t=st.time_input("1차 판단 시각",time(15,30))
    ma_days=st.number_input("이동평균 기간",5,40,20)
    discount=st.number_input("이평선 대비 할인율(%)",0.0,5.0,0.5,0.1)
    low_lookback=st.number_input("최근 저가 기간(거래일)",2,20,5)
    low_band=st.number_input("최근 저가 허용폭(%)",0.0,2.0,0.2,0.1)
    reversal=st.checkbox("전일 15:30보다 상승한 날만 1차 진입",True)
    st.divider()
    levels=st.number_input("최대 차수",2,10,7)
    tranche_amt=st.number_input("차수별 금액(원)",10000,value=100000,step=10000)
    gap_pct=st.number_input("추가매수 간격(%)",0.1,5.0,0.5,0.1)
    take_pct=st.number_input("차수별 목표수익(%)",0.1,5.0,0.5,0.1)
    cost_pct=st.number_input("왕복 비용·스프레드(%)",0.0,2.0,0.10,0.01)

if st.button("🚀 V4 백테스트 실행",type="primary",use_container_width=True):
    try:
        with st.spinner("5분봉 다운로드 및 7차 체결 시뮬레이션 중..."):
            bars=load_fx(period,interval)
            sig=daily_signal_table(bars,entry_t,ma_days,discount,low_lookback,low_band,reversal)
            trades,cycles,realized,unreal,max_cap,last=simulate(bars,sig,tranche_amt,levels,gap_pct,take_pct,cost_pct)
            st.session_state.v4=(bars,sig,trades,cycles,realized,unreal,max_cap,last)
    except Exception as e: st.error(f"실행 실패: {e}")

if "v4" in st.session_state:
    bars,sig,trades,cycles,realized,unreal,max_cap,last=st.session_state.v4
    closed=[c for c in cycles if pd.notna(c["end"])]
    total=realized+unreal
    roi=total/max_cap*100 if max_cap else 0
    a,b,c,d=st.columns(4)
    a.metric("1차 신호",f"{int(sig.signal.sum())}회")
    a.metric("완료 사이클",f"{len(closed)}회")
    b.metric("실현손익",f"{realized:,.0f}원")
    b.metric("미실현손익",f"{unreal:,.0f}원")
    c.metric("총 손익",f"{total:,.0f}원")
    c.metric("최대 동시투입",f"{max_cap:,.0f}원")
    d.metric("자본 대비 수익률",f"{roi:+.2f}%")
    d.metric("마지막 환율",f"{last:,.2f}원")

    st.subheader("① 1차 진입 신호")
    sv=sig[["date","close","ma","low_n","ma_ok","low_ok","reversal_ok","signal"]].copy()
    sv.columns=["날짜","15:30환율","이동평균","최근저가","이평할인","저가근접","반전확인","진입신호"]
    st.dataframe(sv.sort_values("날짜",ascending=False),use_container_width=True,hide_index=True)

    st.subheader("② 실제 모의체결 내역")
    if trades.empty: st.info("설정 조건에서 체결된 거래가 없습니다. 할인율 또는 반전 조건을 조정해 보세요.")
    else:
        st.dataframe(trades.sort_values("시각",ascending=False).style.format({"환율":"{:,.3f}","금액":"{:,.0f}","실현손익":"{:,.0f}"}),use_container_width=True,hide_index=True)
        st.download_button("체결내역 CSV 다운로드",trades.to_csv(index=False).encode("utf-8-sig"),"seven_split_v4_trades.csv","text/csv",use_container_width=True)

    st.subheader("③ 사이클 요약")
    cr=[]
    for x in cycles:
        cr.append({"사이클":x["id"],"시작":x["start"],"종료":x["end"],"1차환율":x["entry"],"최대진입차수":x["max_level"],"실현손익":x["pnl"],"상태":"완료" if pd.notna(x["end"]) else "보유중"})
    cdf=pd.DataFrame(cr)
    if not cdf.empty:
        st.dataframe(cdf.style.format({"1차환율":"{:,.3f}","실현손익":"{:,.0f}"}),use_container_width=True,hide_index=True)
        st.download_button("사이클 CSV 다운로드",cdf.to_csv(index=False).encode("utf-8-sig"),"seven_split_v4_cycles.csv","text/csv",use_container_width=True)

    st.subheader("④ 체결 가정과 주의사항")
    st.markdown("""
- 1차 신호는 **이동평균 대비 할인 + 최근 저가 근접 + 전일 대비 반전**으로 생성합니다.
- 2~7차는 1차 환율에서 설정 간격만큼 복리 하락한 가격에 한 번씩 매수합니다.
- 각 차수는 자신의 매수가 대비 목표수익에 도달하면 독립적으로 매도합니다.
- 같은 5분봉에서 매수·매도 가격이 동시에 닿아도, 새로 산 차수는 그 봉에서 즉시 매도하지 않는 보수적 가정을 사용합니다.
- 왕복 비용은 매도 proceeds에서 차감합니다. 실제 은행별 스프레드·우대율·세금은 다를 수 있습니다.
- 60일은 표본이 작습니다. 이 버전은 체결 로직 검증용이며, 최종 판단은 1년 이상 데이터로 다시 검증해야 합니다.
""")
