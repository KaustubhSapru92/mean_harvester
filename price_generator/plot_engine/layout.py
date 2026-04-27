from dash import dcc, html
from .theme import CHART_THEME, FONT, BTN_ACTIVE, BTN_INACTIVE


def build_layout(symbol: str):
    return html.Div(
        style={
            "backgroundColor": CHART_THEME["bg"],
            "minHeight": "100vh",
            "fontFamily": FONT
        },
        children=[
            _topbar(symbol),
            dcc.Tabs(
                id="main-tabs",
                value="tab-price",
                colors={
                    "border": CHART_THEME["border"],
                    "primary": "#2962ff",
                    "background": CHART_THEME["bg"]
                },
                children=[
                    dcc.Tab(
                        label="Price",
                        value="tab-price",
                        children=[_toolbar(), _price_chart()],
                        style={"color": CHART_THEME["text"]},
                        selected_style={"color": "#d1d4dc",
                                        "backgroundColor": CHART_THEME["bg"],
                                        "borderTop": "2px solid #2962ff"}
                    ),
                    dcc.Tab(
                        label="Price vs VWAP",
                        value="tab-vwap",
                        children=[_vwap_chart()],
                        style={"color": CHART_THEME["text"]},
                        selected_style={"color": "#d1d4dc",
                                        "backgroundColor": CHART_THEME["bg"],
                                        "borderTop": "2px solid #2962ff"}
                    ),
                    dcc.Tab(
                        label="Rolling Dev Z",
                        value="tab-devz",
                        children=[_devz_chart()],
                        style={"color": CHART_THEME["text"]},
                        selected_style={"color": "#d1d4dc",
                                        "backgroundColor": CHART_THEME["bg"],
                                        "borderTop": "2px solid #2962ff"}
                    ),
                ]
            ),
            dcc.Interval(id="interval", interval=1000, n_intervals=0),
            dcc.Store(id="chart-type-store", data="line"),
            dcc.Store(id="ma-store", data=False),
            dcc.Store(id="version-store", data=0),
        ]
    )


def _price_chart():
    return dcc.Graph(
        id="trade-chart",
        config={"displayModeBar": False},
        style={"height": "calc(100vh - 160px)"}
    )


def _vwap_chart():
    return dcc.Graph(
        id="vwap-chart",
        config={"displayModeBar": False},
        style={"height": "calc(100vh - 120px)"}
    )


def _devz_chart():
    return dcc.Graph(
        id="devz-chart",
        config={"displayModeBar": False},
        style={"height": "calc(100vh - 120px)"}
    )

def _topbar(symbol: str):
    return html.Div(
        style={
            "backgroundColor": CHART_THEME["bg"],
            "padding": "14px 20px 10px",
            "borderBottom": f"1px solid {CHART_THEME['border']}",
            "display": "flex",
            "alignItems": "center",
            "gap": "24px",
            "flexWrap": "wrap"
        },
        children=[
            html.Span(symbol, style={"color": "#d1d4dc", "fontSize": "18px", "fontWeight": "600", "letterSpacing": "0.3px"}),
            html.Span(id="cur-price", style={"color": "#d1d4dc", "fontSize": "22px", "fontWeight": "500", "fontVariantNumeric": "tabular-nums"}),
            html.Span(id="cur-change", style={"fontSize": "13px", "fontWeight": "500"}),
            html.Span(id="market-badge"),
        ]
    )


def _toolbar():
    return html.Div(
        style={
            "padding": "8px 20px",
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "borderBottom": f"1px solid {CHART_THEME['border']}"
        },
        children=[
            html.Button("Line",   id="btn-line",   n_clicks=0, style=BTN_ACTIVE),
            html.Button("Candle", id="btn-candle", n_clicks=0, style=BTN_INACTIVE),
            html.Div(style={"width": "1px", "height": "16px", "background": "#2a2e39", "margin": "0 4px"}),
            html.Button("MA 20",  id="btn-ma",     n_clicks=0, style=BTN_INACTIVE),
        ]
    )