from dash import Input, Output, State, ctx
from .theme import CHART_THEME, BTN_ACTIVE, BTN_INACTIVE, BADGE_OPEN, BADGE_CLOSED
from .chart_builder import ChartBuilder

builder = ChartBuilder()


def register_callbacks(app, queue_consumer):

    # =====================
    # Chart type toggle
    # =====================
    @app.callback(
        Output("chart-type-store", "data"),
        Output("btn-line", "style"),
        Output("btn-candle", "style"),
        Input("btn-line", "n_clicks"),
        Input("btn-candle", "n_clicks"),
        State("chart-type-store", "data"),
        prevent_initial_call=True
    )
    def toggle_chart_type(line_clicks, candle_clicks, current_type):
        triggered = ctx.triggered_id
        if triggered == "btn-line":
            return "line", BTN_ACTIVE, BTN_INACTIVE
        elif triggered == "btn-candle":
            return "candle", BTN_INACTIVE, BTN_ACTIVE
        return current_type, BTN_ACTIVE if current_type == "line" else BTN_INACTIVE, BTN_INACTIVE if current_type == "line" else BTN_ACTIVE

    # =====================
    # MA toggle
    # =====================
    @app.callback(
        Output("ma-store", "data"),
        Output("btn-ma", "style"),
        Input("btn-ma", "n_clicks"),
        State("ma-store", "data"),
        prevent_initial_call=True
    )
    def toggle_ma(n_clicks, show_ma):
        new_state = not show_ma
        return new_state, BTN_ACTIVE if new_state else BTN_INACTIVE

    # =====================
    # Chart update
    # =====================
    @app.callback(
        Output("trade-chart", "figure"),
        Output("cur-price", "children"),
        Output("cur-change", "children"),
        Output("cur-change", "style"),
        Output("market-badge", "children"),
        Output("market-badge", "style"),
        Input("interval", "n_intervals"),
        State("chart-type-store", "data"),
        State("ma-store", "data"),
    )
    def update_chart(n, chart_type, show_ma):
        ticks = queue_consumer.get_ticks()

        if not ticks:
            return builder.empty_figure(), "-", "-", {}, "NO DATA", {}

        cur, change, change_pct = builder.get_price_stats(ticks)

        price_str = f"₹{cur:,.2f}"
        is_positive = change >= 0
        change_str = f"{'+'if is_positive else ''}{change:.2f} ({'+'if is_positive else ''}{change_pct:.2f}%)"
        change_style = {
            "fontSize": "13px",
            "fontWeight": "500",
            "color": CHART_THEME["up_color"] if is_positive else CHART_THEME["down_color"]
        }

        fig = builder.build(ticks, chart_type, show_ma)

        # TODO: wire real market status from StockEngine when available
        badge_text = "CLOSED"
        badge_style = BADGE_CLOSED

        return fig, price_str, change_str, change_style, badge_text, badge_style