import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from functools import lru_cache


HEADER_FONT_SIZE       = 32
LABEL_FONT_SIZE        = 18
DROPDOWN_FONT_SIZE     = 16
PLOT_TITLE_FONT_SIZE   = 20
AXIS_TITLE_FONT_SIZE   = 16
DEFAULT_FONT_SIZE      = 14
LEGEND_FONT_SIZE       = 14
FIGURE_HEIGHT          = 600
WINDOW_DAYS            = 7
RL_FIGURE_HEIGHT = 900
WINDOW_HOURS     = 3

def load_real_data(file_path='./data/shortdata.csv'):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def generate_rnn_gan_data():
    df = pd.read_csv('data/short_rnn_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def generate_tcn_gan_data():
    df = pd.read_csv('data/short_tcn_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


def generate_bootstrap_data():
    df = pd.read_csv('./data/short_bootstrap_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@lru_cache(maxsize=4)
def load_decisions(kind: str) -> pd.DataFrame:
    file_map = {
        'real':      './data/real_decisions.csv',
        'rnn_gan':   './data/rnn_decisions.csv',
        'tcn_gan':   './data/tcn_decisions.csv',
        'bootstrap': './data/bootstrap_decisions.csv'
    }
    df = pd.read_csv(file_map[kind])
    df['date'] = pd.to_datetime(df['date'])
    return df

def compute_balances(dec_df: pd.DataFrame,
                     initial_cash: float = 8000.0) -> pd.Series:

    cash      = np.full(len(dec_df), initial_cash, dtype='float64')
    holdings  = np.zeros(len(dec_df), dtype='int32')

    side = dec_df['decision'].map({'Buy': 1, 'Sell': -1}).fillna(0).to_numpy()
    price = dec_df['close'].to_numpy()

    holdings = side.cumsum()

    cash_change = np.where(side != 0, -side * price, 0)
    cash = initial_cash + cash_change.cumsum()

    balance = cash + holdings * price
    return pd.Series(balance, index=dec_df.index, name='balance')


RANGE_BREAKS = [
    dict(bounds=["sat", "mon"]),
    dict(bounds=[17, 9.5], pattern="hour")
]

def create_rl_animation(price_df: pd.DataFrame,
                        decision_df: pd.DataFrame,
                        max_frames: int = 1_000) -> go.Figure:
    last_y_min, last_y_max = None, None

    def window_range(start: pd.Timestamp, end: pd.Timestamp):
        nonlocal last_y_min, last_y_max

        w = price_df[(price_df['date'] >= start) & (price_df['date'] <= end)]

        if w.empty:
            return last_y_min, last_y_max

        low, high = w['low'].min(), w['high'].max()

        if high == low:
            pad = 0.01 * high if high else 1
        else:
            pad = max(0.10 * (high - low), 0.02 * high)

        last_y_min, last_y_max = low - pad, high + pad
        return last_y_min, last_y_max

    def balance_annotation(x: float, y: float, text: str):
        return dict(
            xref='paper', yref='paper', x=x, y=y,
            text=text, showarrow=False,
            font=dict(size=16, color='gold'),
            bgcolor='rgba(50,50,50,0.7)', borderpad=4
        )
    prev_close = price_df['close'].shift(1).fillna(price_df['open'])
    candle = go.Candlestick(
        x=price_df['date'], open=prev_close,
        high=price_df['high'], low=price_df['low'], close=price_df['close'],
        increasing_line_color='green', decreasing_line_color='red',
        increasing_fillcolor='green', decreasing_fillcolor='red',
        name='Price'
    )

    # seeds
    buy_seed  = go.Scatter(mode='markers', x=[], y=[],
                           marker_symbol='arrow-up', marker_color='green',
                           marker_size=12, name='Buy')
    sell_seed = go.Scatter(mode='markers', x=[], y=[],
                           marker_symbol='arrow-down', marker_color='red',
                           marker_size=12, name='Sell')

    #frame
    frame_step = max(1, len(decision_df) // max_frames)

    frames = []
    for i in range(frame_step, len(decision_df) + 1, frame_step):
        sub = decision_df.iloc[:i]

        buy_pts  = sub[sub['decision'] == 'Buy']
        sell_pts = sub[sub['decision'] == 'Sell']

        buy_trace = go.Scatter(
            x=buy_pts['date'],
            y=buy_pts['close'],
            mode='markers', marker_symbol='arrow-up',
            marker_color='green', marker_size=12
        )
        sell_trace = go.Scatter(
            x=sell_pts['date'],
            y=sell_pts['close'],
            mode='markers', marker_symbol='arrow-down',
            marker_color='red', marker_size=12
        )

        end_time = sub['date'].iloc[-1]
        start_time = end_time - pd.Timedelta(hours=WINDOW_HOURS)
        y_min, y_max = window_range(start_time, end_time)
        current_balance = decision_df['balance'].iloc[i - 1]
        frames.append(dict(
            name=str(i),
            data=[buy_trace, sell_trace],
            traces=[1, 2],
            layout=dict(
                xaxis=dict(range=[start_time, end_time],
                           type='date',
                           rangebreaks=RANGE_BREAKS),
                yaxis=dict(range=[y_min, y_max]),
        annotations=[balance_annotation(0.98, 0.95,
                                         f'Balance $ {current_balance:,.0f}')]
            )
        ))

    #viewport
    init_start = price_df['date'].iloc[0]
    init_end = init_start + pd.Timedelta(hours=WINDOW_HOURS)
    y_min, y_max = window_range(init_start, init_end)
    init_balance = decision_df['balance'].iloc[0]
    init_annot = [balance_annotation(0.98, 0.95,
                                     f'Balance $ {init_balance:,.0f}')]
    layout = go.Layout(
        title='Animated RL Strategy — 3-Hour Rolling Window',
        annotations=init_annot,
        title_font={'size': PLOT_TITLE_FONT_SIZE},
        xaxis=dict(title='Date', type='date',
                   range=[init_start, init_end],
                   rangeslider={'visible': False},
                   rangebreaks=RANGE_BREAKS),
        yaxis=dict(title='Price', range=[y_min, y_max]),
        yaxis_title_font={'size': AXIS_TITLE_FONT_SIZE},
        font={'size': DEFAULT_FONT_SIZE},
        height=RL_FIGURE_HEIGHT,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, font={'size': LEGEND_FONT_SIZE}),
        updatemenus=[dict(
            type='buttons', showactive=False,
            x=0.5, y=-0.25, xanchor='center', yanchor='top',
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None,
                      dict(frame=dict(duration=350, redraw=True),
                           fromcurrent=True,
                           transition=dict(duration=0))]
            )]
        )]
    )

    return go.Figure(data=[candle, buy_seed, sell_seed],
                     frames=frames,
                     layout=layout)


@lru_cache(maxsize=4)
def load_and_prep(kind: str):
    real = load_real_data()
    if kind == 'real':
        synth = real
    elif kind == 'rnn_gan':
        synth = generate_rnn_gan_data()
    elif kind == 'tcn_gan':
        synth = generate_tcn_gan_data()
    else:
        synth = generate_bootstrap_data()
    return real, synth



# Dash App
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'RL Trading Strategy'

data_options = [
    {'label': 'Real Data',  'value': 'real'},
    {'label': 'RNN_GAN',    'value': 'rnn_gan'},
    {'label': 'TCN_GAN',    'value': 'tcn_gan'},
    {'label': 'Bootstrap',  'value': 'bootstrap'}
]

app.layout = html.Div([
    html.H1('Synthetic Data-Driven Optimization of RL-Based Scalping Trading',
            style={'textAlign': 'center',
                   'fontSize': f'{HEADER_FONT_SIZE}px'}),
    html.Div([
        html.Label('Select Data Type:',
                   style={'fontSize': f'{LABEL_FONT_SIZE}px'}),
        dcc.Dropdown(id='synthetic-type', options=data_options,
                     value='rnn_gan', clearable=False,
                     style={'fontSize': f'{DROPDOWN_FONT_SIZE}px'})
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Br(),
    html.Div([
        dcc.Graph(id='graph-ohlc',   config={'displayModeBar': False}),
        dcc.Graph(id='graph-cum-log', config={'displayModeBar': False}),
        dcc.Graph(id='graph-rl',     config={'displayModeBar': False})
    ])
], style={'width': '90%', 'margin': 'auto',
          'fontFamily': 'Roboto, Arial, sans-serif',
          'color': '#2f4f4f'})

@app.callback(
    Output('graph-ohlc', 'figure'),
    Output('graph-cum-log', 'figure'),
    Output('graph-rl', 'figure'),
    Input('synthetic-type', 'value')
)


def update_graphs(kind: str):
    real_df, synth_df = load_and_prep(kind)

    cols = ['open', 'high', 'low', 'close']
    fig_returns = make_subplots(rows=2, cols=2,
                                subplot_titles=[f'{c.capitalize()} Return'
                                                for c in cols])
    pos = {0: (1, 1), 1: (1, 2), 2: (2, 1), 3: (2, 2)}
    for i, col in enumerate(cols):
        r, c_pos = pos[i]
        real_r = np.log(real_df[col] / real_df[col].shift(1)).dropna()
        lo, hi = real_r.min(), real_r.max()
        lo += 0.045; hi -= 0.04
        xb = {'start': lo, 'end': hi, 'size': (hi - lo) / 100}
        fig_returns.add_trace(
            go.Histogram(x=real_r, opacity=0.6,
                         xbins=xb, name='Real Return', marker_color='rgba(31,119,180,0.8)',
                         showlegend=(i == 0)),
            row=r, col=c_pos
        )
        if kind != 'real':
            synth_r = np.log(synth_df[col] / synth_df[col].shift(1)).dropna()
            fig_returns.add_trace(
                go.Histogram(x=synth_r, opacity=0.6,
                             xbins=xb, name='Synthetic Return',  marker_color='rgba(255,127,14,0.8)',
                             showlegend=(i == 0)),
                row=r, col=c_pos
            )
        fig_returns.update_xaxes(range=[lo, hi], row=r, col=c_pos)
    fig_returns.update_layout(
        height=FIGURE_HEIGHT, barmode='overlay',
        title_text=('Return Distributions' if kind == 'real'
                    else 'Return Distributions (Real vs Synthetic)'),
        title_font={'size': PLOT_TITLE_FONT_SIZE},
        legend={'orientation': 'h', 'x': 0.43, 'y': 1.15, 'font':{'size': LEGEND_FONT_SIZE}}
    )
    fig_returns.update_yaxes(range=[-50, None])

    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(x=real_df['date'], y=real_df['cumulative_log_return'],
                   mode='lines', name='Real',
                   line={'color':'rgba(31,119,180,0.8)','width': 1.5,'dash':'solid'}))
    if kind != 'real':
        fig_cum.add_trace(
            go.Scatter(x=synth_df['date'],
                       y=synth_df['cumulative_log_return'],
                       mode='lines', name='Synthetic',
                       line={'color':'rgba(255,127,14,0.8)','width':1.5,'dash':'solid'}))
    fig_cum.update_layout(
        title='Cumulative Log Returns',
        xaxis_title='Date', yaxis_title='Cumulative Log Return',
        title_font={'size': PLOT_TITLE_FONT_SIZE},
        font={'size': DEFAULT_FONT_SIZE},
        legend={'x': 0.02, 'y': 0.98, 'font':{'size': LEGEND_FONT_SIZE}}
    )

    decisions_df = load_decisions(kind)
    decisions_df['balance'] = compute_balances(decisions_df)
    fig_rl = create_rl_animation(decisions_df, decisions_df)

    return fig_returns, fig_cum, fig_rl

if __name__=='__main__':
    app.run(debug=False)
