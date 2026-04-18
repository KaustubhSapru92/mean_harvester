import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .theme import CHART_THEME


class ChartBuilder:

    def build(self, ticks: list, chart_type: str, show_ma: bool) -> go.Figure:
        if not ticks:
            return self.empty_figure()

        timestamps = [t["timestamp"] for t in ticks]
        prices = [t["price"] for t in ticks]

        fig = make_subplots(rows=1, cols=1)

        if chart_type == "line":
            self._add_line(fig, timestamps, prices)
        else:
            self._add_candlestick(fig, timestamps, ticks, prices)

        if show_ma and len(prices) >= 20:
            self._add_ma(fig, timestamps, prices, period=20)

        self._apply_layout(fig)
        return fig

    def get_price_stats(self, ticks: list):
        prices = [t["price"] for t in ticks]
        cur = prices[-1]
        open_price = ticks[0].get("Open", prices[0]) if ticks else 0
        change = cur - open_price
        change_pct = (change / open_price) * 100 if open_price else 0
        return cur, change, change_pct

    # =====================
    # Traces
    # =====================
    def _add_line(self, fig, timestamps, prices):
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode="lines",
            line=dict(color=CHART_THEME["price_line"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(41,98,255,0.08)",
            hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>",
            name="Price"
        ))

    def _add_candlestick(self, fig, timestamps, ticks, prices):
        if ticks and all(k in ticks[0] for k in ("Open", "High", "Low", "Close")):
            opens = [t["Open"] for t in ticks]
            highs = [t["High"] for t in ticks]
            lows = [t["Low"] for t in ticks]
            closes = [t["Close"] for t in ticks]
        else:
            opens, highs, lows, closes = self._build_ohlc(prices)
        fig.add_trace(go.Candlestick(
            x=timestamps,
            open=opens, high=highs, low=lows, close=closes,
            increasing_line_color=CHART_THEME["up_color"],
            decreasing_line_color=CHART_THEME["down_color"],
            name="Price"
        ))

    def _add_ma(self, fig, timestamps, prices, period: int):
        ma = self._calc_ma(prices, period)
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=ma,
            mode="lines",
            line=dict(color=CHART_THEME["ma_line"], width=1.2),
            hovertemplate=f"MA{period}: ₹%{{y:,.2f}}<extra></extra>",
            name=f"MA {period}"
        ))

    # =====================
    # Layout
    # =====================
    def _apply_layout(self, fig):
        fig.update_layout(
            paper_bgcolor=CHART_THEME["paper_bg"],
            plot_bgcolor=CHART_THEME["bg"],
            margin=dict(l=10, r=60, t=10, b=30),
            xaxis=dict(
                showgrid=True,
                gridcolor=CHART_THEME["grid"],
                tickfont=dict(color=CHART_THEME["text"], size=11, family="Inter"),
                linecolor=CHART_THEME["border"],
                rangeslider=dict(visible=False),
                showspikes=True,
                spikecolor="#434651",
                spikethickness=1,
                spikedash="solid",
                spikemode="across"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=CHART_THEME["grid"],
                tickfont=dict(color=CHART_THEME["text"], size=11, family="Inter"),
                linecolor=CHART_THEME["border"],
                side="right",
                tickprefix="₹",
                showspikes=True,
                spikecolor="#434651",
                spikethickness=1,
            ),
            hoverlabel=dict(
                bgcolor=CHART_THEME["tooltip_bg"],
                bordercolor="#2a2e39",
                font=dict(color="#d1d4dc", size=13, family="Inter")
            ),
            showlegend=False,
            dragmode="pan",
            hovermode="x unified"
        )

    def empty_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=CHART_THEME["paper_bg"],
            plot_bgcolor=CHART_THEME["bg"],
            margin=dict(l=10, r=60, t=10, b=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(
                text="Waiting for data...",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#787b86", size=14, family="Inter")
            )]
        )
        return fig

    # =====================
    # Helpers
    # =====================
    def _calc_ma(self, prices: list, period: int) -> list:
        return [
            None if i < period - 1
            else round(sum(prices[i - period + 1:i + 1]) / period, 2)
            for i in range(len(prices))
        ]

    def _build_ohlc(self, prices: list):
        opens, highs, lows, closes = [], [], [], []
        for i in range(len(prices)):
            o = prices[i - 1] if i > 0 else prices[i]
            c = prices[i]
            h = max(o, c) + abs(c - o) * 0.3
            l = min(o, c) - abs(c - o) * 0.3
            opens.append(round(o, 2))
            highs.append(round(h, 2))
            lows.append(round(l, 2))
            closes.append(round(c, 2))
        return opens, highs, lows, closes
