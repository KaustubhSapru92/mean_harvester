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

    def build_signal(self, dev_z: dict) -> go.Figure:
        fig = go.Figure()

        colors = {"DEV_20": CHART_THEME["price_line"], "DEV_50": CHART_THEME["ma_line"]}

        for key, values in dev_z.items():
            if values is None:
                continue
            fig.add_trace(go.Scatter(
                y=values, mode="lines",
                line=dict(color=colors.get(key), width=1.2),
                name=key
            ))

        # ±2σ threshold lines
        for y, label in [(2, "+2σ"), (-2, "-2σ")]:
            fig.add_hline(y=y, line_dash="dash", line_color="#434651",
                          annotation_text=label,
                          annotation_font_color=CHART_THEME["text"])

        self._apply_signal_layout(fig)
        return fig

    def _apply_signal_layout(self, fig):
        fig.update_layout(paper_bgcolor=CHART_THEME["paper_bg"],
                          plot_bgcolor=CHART_THEME["bg"],
                          margin=dict(l=10, r=60, t=10, b=30),
                          xaxis=dict(showgrid=True, gridcolor=CHART_THEME["grid"],
                                     tickfont=dict(color=CHART_THEME["text"], size=11)),
                          yaxis=dict(showgrid=True, gridcolor=CHART_THEME["grid"],
                                     tickfont=dict(color=CHART_THEME["text"], size=11),
                                     side="right"),
                          showlegend=True,
                          legend=dict(font=dict(color=CHART_THEME["text"])))

    def build_price_vs_vwap(self, vwap_df) -> go.Figure:
        if vwap_df is None or vwap_df.empty:
            return self.empty_figure()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=vwap_df.index,
            y=vwap_df["Close"],
            mode="lines",
            line=dict(color=CHART_THEME["price_line"], width=1.5),
            name="Close"
        ))

        colors = {20: CHART_THEME["ma_line"], 50: "#a78bfa"}
        for w in [20, 50]:
            col = f"VWAP_{w}"
            if col in vwap_df.columns:
                fig.add_trace(go.Scatter(
                    x=vwap_df.index,
                    y=vwap_df[col],
                    mode="lines",
                    line=dict(color=colors[w], width=1.2),
                    name=f"VWAP {w}"
                ))

        self._apply_layout(fig)
        return fig

    def build_dev_z(self, stability_map: dict) -> go.Figure:
        if not stability_map:
            return self.empty_figure()

        fig = go.Figure()

        colors = {20: CHART_THEME["price_line"], 50: CHART_THEME["ma_line"]}
        for w, stability_df in stability_map.items():
            if stability_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=stability_df.index,
                y=stability_df["dev_z"],
                mode="lines",
                line=dict(color=colors.get(w), width=1.2),
                name=f"DEV_Z {w}"
            ))

        for y, label in [(2, "+2σ"), (-2, "-2σ")]:
            fig.add_hline(
                y=y,
                line_dash="dash",
                line_color="#434651",
                annotation_text=label,
                annotation_font_color=CHART_THEME["text"]
            )

        self._apply_signal_layout(fig)
        return fig
