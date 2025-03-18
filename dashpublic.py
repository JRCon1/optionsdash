# Import additional required packages
from dash import callback_context
import dash_bootstrap_components as dbc  # For improved UI components
from dash.exceptions import PreventUpdate
import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go  # Add this import at the top with other imports
import io

# Define common styles
COMMON_STYLES = {
    'container': {
        'padding': '20px',
        'backgroundColor': '#282a36',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
        'marginBottom': '20px',
        'border': '1px solid #44475a'
    },
    'header': {
        'textAlign': 'center',
        'fontWeight': 'bold',
        'marginBottom': '20px',
        'color': '#f8f8f2',
        'fontSize': '28px',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'letterSpacing': '0.5px'
    },
    'subheader': {
        'textAlign': 'center',
        'fontWeight': 'bold',
        'marginBottom': '15px',
        'color': '#bd93f9',
        'fontSize': '22px',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'letterSpacing': '0.3px'
    },
    'input_group': {
        'marginBottom': '15px'
    },
    'label': {
        'fontWeight': 'bold',
        'marginBottom': '5px',
        'color': '#f8f8f2',
        'fontSize': '14px',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'letterSpacing': '0.2px'
    },
    'value': {
        'color': '#50fa7b',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'fontSize': '16px'
    },
    'table': {
        'margin': 'auto',
        'border': '1px solid #44475a',
        'borderCollapse': 'collapse',
        'width': '100%'
    },
    'table_header': {
        'fontWeight': 'bold',
        'padding': '8px',
        'textAlign': 'center',
        'backgroundColor': '#44475a',
        'border': '1px solid #6272a4',
        'color': '#f8f8f2',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'fontSize': '14px'
    },
    'table_cell': {
        'padding': '8px',
        'textAlign': 'center',
        'border': '1px solid #44475a',
        'color': '#f8f8f2',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'fontSize': '13px'
    },
    'price_container': {
        'backgroundColor': '#44475a',
        'padding': '15px 20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between'
    },
    'price_label': {
        'color': '#f8f8f2',
        'fontSize': '16px',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'marginRight': '10px'
    },
    'price_value': {
        'color': '#50fa7b',
        'fontSize': '24px',
        'fontWeight': '600',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'letterSpacing': '0.5px'
    },
    'button_container': {
        'display': 'flex',
        'gap': '10px'
    },
    'download_button': {
        'backgroundColor': '#6272a4',
        'color': '#f8f8f2',
        'border': 'none',
        'padding': '10px 15px',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'fontFamily': '"Fira Code", "Consolas", monospace',
        'fontSize': '14px',
        'transition': 'all 0.3s ease'
    }
}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, 
    external_stylesheets=[
        dbc.themes.DARKLY,
        'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
    ],
    meta_tags=[{'name': 'viewport',
                'content': 'width=device-width, initial-scale=1.0'}]
)

server = app.server

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Options Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {
                background-color: #282a36;
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
            }
            .nav-link.active {
                background-color: #44475a !important;
                color: #f8f8f2 !important;
                font-family: "Fira Code", "Consolas", monospace;
                font-weight: 500;
            }
            .nav-link {
                color: #f8f8f2 !important;
                font-family: "Fira Code", "Consolas", monospace;
                font-weight: 400;
            }
            .nav-tabs .nav-link {
                color: #f8f8f2 !important;
                border-color: #44475a #44475a #282a36;
                font-size: 14px;
                letter-spacing: 0.2px;
            }
            .nav-tabs .nav-link:hover {
                border-color: #6272a4 #6272a4 #282a36;
                color: #bd93f9 !important;
            }
            .nav-tabs .nav-link.active {
                background-color: #44475a !important;
                border-color: #44475a #44475a #282a36;
                color: #bd93f9 !important;
                font-weight: 600;
            }
            .card {
                background-color: #282a36;
                border: 1px solid #44475a;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
                border-color: #6272a4;
            }
            .table-container {
                overflow-x: auto;
                margin: 20px 0;
            }
            .dropdown-container {
                margin-bottom: 15px;
            }
            .Select-control {
                background-color: #44475a;
                border-color: #6272a4;
            }
            .Select-menu-outer {
                background-color: #44475a;
                border-color: #6272a4;
            }
            .Select-option {
                background-color: #44475a;
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 13px;
            }
            .Select-option:hover {
                background-color: #6272a4;
                color: #f8f8f2;
            }
            .Select-value-label {
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 13px;
            }
            .Select-placeholder {
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 13px;
            }
            .Select--multi .Select-value {
                background-color: #6272a4;
                border-color: #6272a4;
            }
            .Select--multi .Select-value-icon {
                border-color: #6272a4;
            }
            .Select--multi .Select-value-icon:hover {
                background-color: #ff5555;
            }
            .form-control {
                background-color: #44475a;
                border-color: #6272a4;
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 13px;
            }
            .form-control:focus {
                background-color: #44475a;
                border-color: #bd93f9;
                color: #f8f8f2;
                box-shadow: 0 0 0 0.2rem rgba(189, 147, 249, 0.25);
            }
            .rc-slider-track {
                background-color: #6272a4;
            }
            .rc-slider-handle {
                border-color: #6272a4;
                background-color: #bd93f9;
            }
            .rc-slider-handle:hover {
                border-color: #bd93f9;
            }
            .rc-slider-handle:active {
                border-color: #bd93f9;
                box-shadow: 0 0 5px #bd93f9;
            }
            .rc-slider-rail {
                background-color: #44475a;
            }
            .rc-slider-mark-text {
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 12px;
            }
            .rc-slider-mark-text-active {
                color: #bd93f9;
            }
            .dash-table-container {
                background-color: #282a36;
            }
            .dash-table-tooltip {
                background-color: #44475a;
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 12px;
            }
            .dash-table-tooltip-header {
                background-color: #6272a4;
                color: #f8f8f2;
                font-weight: 600;
            }
            .dash-table-tooltip-cell {
                color: #f8f8f2;
            }
            .dash-table-tooltip-cell:hover {
                background-color: #6272a4;
            }
            .dash-table-prev-page, .dash-table-next-page {
                color: #f8f8f2;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 13px;
            }
            .dash-table-prev-page:hover, .dash-table-next-page:hover {
                color: #bd93f9;
            }
            .dash-table-pagination-button {
                background-color: #44475a;
                color: #f8f8f2;
                border-color: #6272a4;
                font-family: "Fira Code", "Consolas", monospace;
                font-size: 13px;
            }
            .dash-table-pagination-button:hover {
                background-color: #6272a4;
                color: #f8f8f2;
            }
            .dash-table-pagination-button--current {
                background-color: #6272a4;
                color: #f8f8f2;
                font-weight: 600;
            }
            .current-price {
                textAlign: "right",
                fontSize: "20px",
                fontWeight: "600",
                padding: "10px",
                color: "#50fa7b",
                fontFamily: '"Fira Code", "Consolas", monospace',
                letterSpacing: "0.3px"
            }
            .greek-value {
                color: #50fa7b;
                fontFamily: '"Fira Code", "Consolas", monospace';
                fontSize: "14px";
                fontWeight: "500";
            }
            .greek-label {
                color: #f8f8f2;
                fontFamily: '"Fira Code", "Consolas", monospace';
                fontSize: "12px";
                opacity: 0.8;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ---------------------------
# Helper Pricing Functions (for recalculation)
# ---------------------------
def bsm_price_calc(S, K, T, sigma, opt_type='c', r=0.05):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'c':
        return round(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 4)
    else:
        return round(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 4)

def monte_carlo_price_calc(S, K, T, sigma, opt_type='c', r=0.05, n=10000, q=0):
    if T <= 0 or sigma <= 0:
        return 0.0
    Z = np.random.normal(0, 1, n)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0) if opt_type == 'c' else np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return round(price, 4)

# ---------------------------
# Data Download and Calculation Functions
# ---------------------------
def download_options(ticker_symbol, opt_type='c', max_days=60, lower_moneyness=0.95, upper_moneyness=1.05):
    ticker = yf.Ticker(ticker_symbol)
    underlying_price = ticker.history(period="1d")['Close'].iloc[-1]
    lower_strike, upper_strike = underlying_price * lower_moneyness, underlying_price * upper_moneyness
    relevant_columns = ['strike', 'inTheMoney', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
    filtered_options = pd.DataFrame(columns=relevant_columns + ['expiry'])
    
    for expiry_date_str in ticker.options:
        expiry_date = pd.to_datetime(expiry_date_str)
        days_to_expiry = (expiry_date - datetime.now()).days
        if days_to_expiry <= max_days:
            option_chain = ticker.option_chain(expiry_date_str)
            data = option_chain.calls if opt_type == 'c' else option_chain.puts
            data = data[(data['strike'] >= lower_strike) & (data['strike'] <= upper_strike)].copy()
            data['expiry'] = expiry_date
            if not data.empty:
                data = data[relevant_columns + ['expiry']]
                filtered_options = pd.concat([filtered_options, data], ignore_index=True)
    
    filtered_options['Days to Expiry'] = (pd.to_datetime(filtered_options['expiry']) - datetime.now()).dt.days
    filtered_options['Mid-Point Price'] = round((filtered_options['bid'] + filtered_options['ask']) / 2, 2)
    filtered_options['expiry'] = pd.to_datetime(filtered_options['expiry']).dt.date
    filtered_options['impliedVolatility'] = round(filtered_options['impliedVolatility'], 4)
    filtered_options.rename(columns={'impliedVolatility': 'Implied Vol.'}, inplace=True)
    filtered_options.rename(columns={'inTheMoney': 'ITM'}, inplace=True)
    filtered_options['strike'] = pd.to_numeric(filtered_options['strike'], errors='coerce')

    # Calculate BSM and Monte Carlo prices using the downloaded IV
    def bsm_price(row, option_type=opt_type, r=0.05):
        S = underlying_price
        K = row['strike']
        T = row['Days to Expiry'] / 365
        sigma = row['Implied Vol.']
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'c':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return round(price, 4)
    
    def monte_carlo_price(row, n=10000, r=0.05, q=0):
        S = underlying_price
        K = row['strike']
        T = row['Days to Expiry'] / 365
        sigma = row['Implied Vol.']
        if T <= 0 or sigma <= 0:
            return 0.0
        Z = np.random.normal(0, 1, n)
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoffs = np.maximum(ST - K, 0) if opt_type == 'c' else np.maximum(K - ST, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        return round(price, 4)
    
    filtered_options['BSM Price'] = filtered_options.apply(bsm_price, axis=1)
    filtered_options['Monte Carlo Price'] = filtered_options.apply(monte_carlo_price, axis=1)
    
    # Add the underlying price to each row (for later recalculation)
    filtered_options['Underlying Price'] = underlying_price
    
    return filtered_options

def compute_d1_d2(S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2

def delta(row, S, r=0.05, option_type='c'):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0: 
        return np.nan
    d1, _ = compute_d1_d2(S, K, T, r, sigma)
    return round(norm.cdf(d1) if option_type == 'c' else norm.cdf(d1) - 1, 4)

def theta(row, S, r=0.05, option_type='c'):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0: 
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    theta_val = (-S * N_prime_d1 * sigma / (2 * np.sqrt(T))) + (r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'c' else norm.cdf(-d2)))
    return round(theta_val / 365, 4)

def vega(row, S):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0: 
        return np.nan
    d1, _ = compute_d1_d2(S, K, T, 0.05, sigma)
    return round(S * np.sqrt(T) * norm.pdf(d1) * 0.01, 4)

def rho(row, S, r=0.05, option_type='c'):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0: 
        return np.nan
    _, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'c' else -norm.cdf(-d2)) * 0.01, 4)

def gamma(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, _ = compute_d1_d2(S, K, T, r, sigma)
    gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return round(gamma_val, 4)

def vanna(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vanna_val = norm.pdf(d1) * d2 / sigma
    return round(vanna_val, 4)

def vomma(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vega_val = S * np.sqrt(T) * norm.pdf(d1) * 0.01
    vomma_val = vega_val * (d1 * d2) / sigma
    return round(vomma_val, 4)

def charm(row, S, r=0.05, option_type='c'):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    charm_value = -N_prime_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T)
    return round(charm_value, 4)

def speed(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, _ = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    speed_val = (N_prime_d1 / (S ** 2 * sigma * np.sqrt(T))) * ((d1 / (sigma * np.sqrt(T))) - 1)
    return round(speed_val, 4)

def color(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    color_val = (N_prime_d1 / (2 * S * T * sigma * np.sqrt(T))) * (2 * r * T + 1 - d1 * d2)
    return round(color_val, 4)

def ultima(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vega_val = S * np.sqrt(T) * norm.pdf(d1) * 0.01
    ultima_val = (vega_val * (d1 * d2 - 1) * d1 * d2) / sigma
    return round(ultima_val, 4)

def zomma(row, S, r=0.05):
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['Implied Vol.']
    if sigma <= 0 or T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    zomma_val = gamma_val * (d1 * d2 - 1) / sigma
    return round(zomma_val, 4)

def calculate_payoff_points(selected_options, price_range_percentage=0.2, num_points=100):
    """Calculate payoff points for the selected options."""
    if not selected_options:
        return None, None, None, None, None
    
    # Get the current underlying price from the first option
    current_price = float(selected_options[0].get("Underlying Price", 0))
    
    # Calculate price range
    price_min = current_price * (1 - price_range_percentage)
    price_max = current_price * (1 + price_range_percentage)
    price_points = np.linspace(price_min, price_max, num_points)
    
    total_payoff = np.zeros(num_points)
    breakeven_points = []
    
    # Initialize max profit and loss
    max_profit = float('-inf')
    max_loss = float('inf')
    
    # Calculate individual payoffs and track min/max values
    for option in selected_options:
        strike = float(option.get("strike", 0))
        premium = float(option.get("Mid-Point Price", 0))
        trade_type = option.get("Trade Type", "Buy")
        option_type = option.get("option_type", "c")  # 'c' for call, 'p' for put
        
        # Calculate payoff for each price point
        if option_type == 'c':
            payoff = np.maximum(price_points - strike, 0)
        else:
            payoff = np.maximum(strike - price_points, 0)
            
        # Adjust for premium and trade type
        if trade_type == "Buy":
            payoff = payoff - premium
        else:  # Sell
            payoff = premium - payoff
                
        total_payoff += payoff
    
    # Calculate max profit and loss from the total payoff curve
    max_profit = np.max(total_payoff)
    max_loss = np.min(total_payoff)
    
    # Find breakeven points (where payoff crosses zero)
    for i in range(len(price_points) - 1):
        if (total_payoff[i] <= 0 <= total_payoff[i + 1]) or (total_payoff[i] >= 0 >= total_payoff[i + 1]):
            # Linear interpolation to find more precise breakeven point
            x1, x2 = price_points[i], price_points[i + 1]
            y1, y2 = total_payoff[i], total_payoff[i + 1]
            breakeven = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            breakeven_points.append(round(breakeven, 2))
    
    return price_points, total_payoff, sorted(breakeven_points), max_profit, max_loss

# ---------------------------
# Dash App Initialization and Layout
# ---------------------------
app = dash.Dash(__name__)

# Load ticker symbols from GitHub
tickers_df = pd.read_csv("https://raw.githubusercontent.com/JRCon1/Technical-Analysis-Project/main/Stock%20Data%207-10%20CSV.csv", dtype=str, low_memory=False)
tickers = tickers_df[tickers_df['Options'] == 'Yes']
tickers = tickers['Symbol'].dropna().unique().tolist()

# Define columns for the DataTable (including price and trade type columns)
table_columns = [
    {"name": "strike", "id": "strike"},
    {"name": "ITM", "id": "ITM"},
    {"name": "lastPrice", "id": "lastPrice"},
    {"name": "bid", "id": "bid"},
    {"name": "ask", "id": "ask"},
    {"name": "volume", "id": "volume"},
    {"name": "openInterest", "id": "openInterest"},
    {"name": "Implied Vol.", "id": "Implied Vol."},
    {"name": "expiry", "id": "expiry"},
    {"name": "Days to Expiry", "id": "Days to Expiry"},
    {"name": "Mid-Point Price", "id": "Mid-Point Price"},
    {"name": "BSM Price", "id": "BSM Price"},
    {"name": "Monte Carlo Price", "id": "Monte Carlo Price"},
    {"name": "Delta", "id": "Delta"},
    {"name": "Theta", "id": "Theta"},
    {"name": "Vega", "id": "Vega"},
    {"name": "Rho", "id": "Rho"},
    {"name": "Gamma", "id": "Gamma"},
    {"name": "Vanna", "id": "Vanna"},
    {"name": "Vomma", "id": "Vomma"},
    {"name": "Charm", "id": "Charm"},
    {"name": "Color", "id": "Color"},
    {"name": "Speed", "id": "Speed"},
    {"name": "Ultima", "id": "Ultima"},
    {"name": "Zomma", "id": "Zomma"},
    {"name": "Trade Type", "id": "Trade Type", "editable": True, "presentation": "dropdown"}
]

# First Tab: Options Greeks Dashboard
def get_options_dashboard():
    return html.Div([
        dbc.Container([
            # New header section with improved styling
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Span("Current Price", style=COMMON_STYLES['price_label']),
                                    html.Div(id='current_price', style=COMMON_STYLES['price_value'])
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], style=COMMON_STYLES['price_container'])
                        ], width=8),
                        dbc.Col([
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-file-csv me-2"), "CSV"],
                                    id="btn-csv",
                                    color="success",
                                    className="me-2",
                                    style={'fontSize': '16px', 'fontFamily': '"Fira Code", "Consolas", monospace', 'padding': '10px 20px'}
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-file-excel me-2"), "Excel"],
                                    id="btn-excel",
                                    color="primary",
                                    style={'fontSize': '16px', 'fontFamily': '"Fira Code", "Consolas", monospace', 'padding': '10px 20px'}
                                )
                            ], style={'display': 'flex', 'justifyContent': 'flex-end', 'gap': '10px', 'marginTop': '10px'})
                        ], width=4)
                    ])
                ])
            ], style={'backgroundColor': '#282a36', 'border': '1px solid #44475a', 'marginBottom': '20px'}),
            
            # Rest of the dashboard content
            dbc.Row([
                dbc.Col([
                    html.H1("Options Greeks Dashboard", style=COMMON_STYLES['header'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Ticker", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ticker',
                                        options=[{'label': i, 'value': i} for i in tickers],
                                        value='AAPL',
                                        className='dropdown-container'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Option Type", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='opt_type',
                                        options=[{'label': 'Call', 'value': 'c'}, {'label': 'Put', 'value': 'p'}],
                                        value='c',
                                        className='dropdown-container'
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Max Days to Expiry", style=COMMON_STYLES['label']),
                                    dcc.Slider(
                                        id='max_days',
                                        min=1,
                                        max=180,
                                        step=1,
                                        value=60,
                                        marks={i: str(i) for i in range(0, 181, 30)}
                                    )
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Moneyness Range", style=COMMON_STYLES['label']),
                                    dcc.RangeSlider(
                                        id='moneyness',
                                        min=0.8,
                                        max=1.2,
                                        step=0.01,
                                        value=[0.95, 1.05],
                                        marks={i: str(i) for i in np.arange(0.8, 1.3, 0.1)}
                                    )
                                ])
                            ])
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(className='table-container', children=[
                        dash_table.DataTable(
                            id='options_table',
                            columns=table_columns,
                            row_selectable='multi',
                            selected_rows=[],
                            persistence=True,
                            persisted_props=['selected_rows'],
                            persistence_type='session',
                            editable=True,
                            dropdown={
                                "Trade Type": {
                                    "options": [
                                        {"label": "Buy", "value": "Buy"},
                                        {"label": "Sell", "value": "Sell"}
                                    ]
                                }
                            },
                            page_size=10,
                            style_table={
                                'overflowX': 'auto',
                                'backgroundColor': '#282a36',
                                'border': '1px solid #44475a'
                            },
                            style_cell={
                                'backgroundColor': '#282a36',
                                'color': '#f8f8f2',
                                'textAlign': 'center',
                                'padding': '10px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'fontFamily': '"Fira Code", "Consolas", monospace',
                                'fontSize': '13px',
                                'border': '1px solid #44475a'
                            },
                            style_header={
                                'backgroundColor': '#44475a',
                                'color': '#f8f8f2',
                                'fontWeight': '600',
                                'fontFamily': '"Fira Code", "Consolas", monospace',
                                'fontSize': '14px',
                                'border': '1px solid #6272a4'
                            },
                            style_data={
                                'backgroundColor': '#282a36',
                                'color': '#f8f8f2',
                                'border': '1px solid #44475a'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#21222c'
                                },
                                {
                                    'if': {'filter_query': '{ITM} = true'},
                                    'color': '#50fa7b'
                                },
                                {
                                    'if': {'filter_query': '{ITM} = false'},
                                    'color': '#ff5555'
                                },
                                {
                                    'if': {'column_id': 'Trade Type'},
                                    'backgroundColor': '#44475a',
                                    'color': '#ffffff',
                                    'fontSize': '16px'
                                }
                            ],
                            style_filter={
                                'backgroundColor': '#44475a',
                                'color': '#f8f8f2'
                            }
                        )
                    ])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2("3D Scatter Plot", style=COMMON_STYLES['header']),
                            html.Label("Select Greek for Z-Axis", style=COMMON_STYLES['label']),
                            dcc.Dropdown(
                                id='greek_choice',
                                options=[{'label': greek, 'value': greek} for greek in 
                                         ["Delta", "Theta", "Vega", "Rho", "Gamma", "Vanna", "Vomma", "Charm", "Color", "Speed", "Ultima", "Zomma"]],
                                value="Delta",
                                className='dropdown-container'
                            ),
                            dcc.Graph(id='scatter_3d')
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ])
        ], fluid=True)
    ])

@app.callback(
    Output('current_price', 'children'),
    [Input('ticker', 'value')]
)
def update_current_price(ticker):
    if ticker:
        try:
            price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            return f"${price:,.2f}"
        except Exception as e:
            return "N/A"
    return "N/A"

# Second Tab: Backtesting Area
def get_backtesting_area():
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("Backtesting Area", style=COMMON_STYLES['header'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("IV Adjustment (%)", style=COMMON_STYLES['label']),
                                    dcc.Input(
                                        id='iv_adjustment',
                                        type='number',
                                        placeholder='Enter adjustment (e.g., 5 for +5%)',
                                        min=-100,
                                        max=100,
                                        step=1,
                                        className='form-control'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Price Override", style=COMMON_STYLES['label']),
                                    dcc.Input(
                                        id='price_override',
                                        type='number',
                                        placeholder='Enter price override',
                                        step=0.01,
                                        className='form-control'
                                    )
                                ], width=6)
                            ])
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='backtesting_summary', style=COMMON_STYLES['container'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Payoff Diagram", style=COMMON_STYLES['subheader']),
                            dcc.Graph(id='payoff_chart'),
                            html.Div(id='payoff_metrics', style={
                                'marginTop': '20px',
                                'padding': '10px',
                                'backgroundColor': '#44475a',
                                'borderRadius': '5px',
                                'color': '#f8f8f2'
                            })
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ])
        ], fluid=True)
    ])

# Add new function for Strategy Templates tab
def get_strategy_templates():
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("Strategy Templates", style=COMMON_STYLES['header'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Select Strategy", style=COMMON_STYLES['label']),
                            dcc.Dropdown(
                                id='strategy_type',
                                options=[
                                    {'label': 'Iron Condor', 'value': 'iron_condor'},
                                    {'label': 'Butterfly', 'value': 'butterfly'},
                                    {'label': 'Calendar Spread', 'value': 'calendar'}
                                ],
                                value='iron_condor',
                                className='dropdown-container'
                            )
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Strategy Builder", style=COMMON_STYLES['subheader']),
                            html.Div([
                                html.P("An Iron Condor is a neutral options strategy that profits from low volatility and time decay. It consists of four options:",
                                    style={'color': '#f8f8f2', 'marginBottom': '15px'}),
                                html.Ul([
                                    html.Li("Sell an out-of-the-money call (short call)",
                                        style={'color': '#f8f8f2'}),
                                    html.Li("Buy a higher strike call (long call)",
                                        style={'color': '#f8f8f2'}),
                                    html.Li("Sell an out-of-the-money put (short put)",
                                        style={'color': '#f8f8f2'}),
                                    html.Li("Buy a lower strike put (long put)",
                                        style={'color': '#f8f8f2'})
                                ], style={'marginBottom': '20px'})
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Ticker", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ic_ticker',
                                        options=[{'label': i, 'value': i} for i in tickers],
                                        value='AAPL',
                                        className='dropdown-container'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Expiration", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ic_expiry',
                                        className='dropdown-container'
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Short Call Strike", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ic_short_call',
                                        className='dropdown-container'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Long Call Strike", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ic_long_call',
                                        className='dropdown-container'
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Short Put Strike", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ic_short_put',
                                        className='dropdown-container'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Long Put Strike", style=COMMON_STYLES['label']),
                                    dcc.Dropdown(
                                        id='ic_long_put',
                                        className='dropdown-container'
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Width (points between strikes)", style=COMMON_STYLES['label']),
                                    dcc.Slider(
                                        id='ic_width',
                                        min=1,
                                        max=20,
                                        step=1,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 21, 2)}
                                    )
                                ])
                            ])
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Strategy Analysis", style=COMMON_STYLES['subheader']),
                            html.Div(id='ic_analysis', style={
                                'marginTop': '20px',
                                'padding': '10px',
                                'backgroundColor': '#44475a',
                                'borderRadius': '5px',
                                'color': '#f8f8f2'
                            }),
                            dcc.Graph(id='ic_payoff_chart')
                        ])
                    ], style=COMMON_STYLES['container'])
                ])
            ])
        ], fluid=True)
    ])

# Main Layout with Tabs
app.layout = html.Div([
    dcc.Download(id="download-dataframe-csv"),
    dcc.Download(id="download-dataframe-excel"),
    dbc.Tabs([
        dbc.Tab(label='Options Greeks Dashboard', tab_id='tab-1', children=get_options_dashboard()),
        dbc.Tab(label='Backtesting Area', tab_id='tab-2', children=get_backtesting_area()),
        dbc.Tab(label='Strategy Templates', tab_id='tab-3', children=get_strategy_templates())
    ], id='tabs', active_tab='tab-1')
])

# ---------------------------
# Callbacks
# ---------------------------
# Update Options Table Data
@app.callback(
    Output('options_table', 'data'),
    [Input('ticker', 'value'),
     Input('opt_type', 'value'),
     Input('max_days', 'value'),
     Input('moneyness', 'value')]
)
def update_options(ticker, opt_type, max_days, moneyness):
    df = download_options(ticker, opt_type, max_days, moneyness[0], moneyness[1])
    if df.empty:
        return []
    
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    df['Delta'] = df.apply(lambda row: delta(row, S, option_type=opt_type), axis=1)
    df['Theta'] = df.apply(lambda row: theta(row, S, option_type=opt_type), axis=1)
    df['Vega'] = df.apply(lambda row: vega(row, S), axis=1)
    df['Rho'] = df.apply(lambda row: rho(row, S, option_type=opt_type), axis=1)
    df['Gamma'] = df.apply(lambda row: gamma(row, S), axis=1)
    df['Vanna'] = df.apply(lambda row: vanna(row, S), axis=1)
    df['Vomma'] = df.apply(lambda row: vomma(row, S), axis=1)
    df['Charm'] = df.apply(lambda row: charm(row, S, option_type=opt_type), axis=1)
    df['Color'] = df.apply(lambda row: color(row, S, r=0.05), axis=1)
    df['Speed'] = df.apply(lambda row: speed(row, S, r=0.05), axis=1)
    df['Ultima'] = df.apply(lambda row: ultima(row, S, r=0.05), axis=1)
    df['Zomma'] = df.apply(lambda row: zomma(row, S, r=0.05), axis=1)
    
    df['Trade Type'] = 'Buy'
    return df.to_dict('records')

# Update 3D Scatter Plot
@app.callback(
    Output('scatter_3d', 'figure'),
    [Input('ticker', 'value'),
     Input('opt_type', 'value'),
     Input('max_days', 'value'),
     Input('moneyness', 'value'),
     Input('greek_choice', 'value')]
)
def update_scatter_3d(ticker, opt_type, max_days, moneyness, greek_choice):
    df = download_options(ticker, opt_type, max_days, moneyness[0], moneyness[1])
    if df.empty:
        return px.scatter_3d(title="No Data Available")
    
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    df['Delta'] = df.apply(lambda row: delta(row, S, option_type=opt_type), axis=1)
    df['Theta'] = df.apply(lambda row: theta(row, S, option_type=opt_type), axis=1)
    df['Vega'] = df.apply(lambda row: vega(row, S), axis=1)
    df['Rho'] = df.apply(lambda row: rho(row, S, option_type=opt_type), axis=1)
    df['Gamma'] = df.apply(lambda row: gamma(row, S), axis=1)
    df['Vanna'] = df.apply(lambda row: vanna(row, S), axis=1)
    df['Vomma'] = df.apply(lambda row: vomma(row, S), axis=1)
    
    df['plot_color'] = df['ITM'].apply(lambda x: 'In the Money' if x else 'Out of the Money')
    
    fig = px.scatter_3d(
        df,
        x='strike',
        y='Days to Expiry',
        z=greek_choice,
        color='plot_color',
        color_discrete_map={'In the Money': 'green', 'Out of the Money': 'red'},
        title=f'3D Scatter: {greek_choice} vs. Strike and DTE ({ticker})'
    )
    
    # Update legend title
    fig.update_layout(
        legend_title_text='Option Status',
        showlegend=True
    )
    
    return fig

# Backtesting Callback: Aggregate Selected Options, Recalculate Prices with IV Adjustment, and Display a Clean Summary
# Backtesting Callback: Aggregate Selected Options, Recalculate Prices & Greeks with IV Adjustment, and Display a Clean Summary
@app.callback(
    Output('backtesting_summary', 'children'),
    [Input('options_table', 'data'),
     Input('options_table', 'selected_rows'),
     Input('iv_adjustment', 'value'),
     Input('price_override', 'value')]
)
def update_backtesting_summary(data, selected_rows, iv_adjustment, price_override):
    if not data or selected_rows is None or len(selected_rows) == 0:
        return html.Div("No options selected.", style={
            "fontFamily": "Helvetica Neue, Arial, sans-serif",
            "fontSize": "18px",
            "fontWeight": "bold",
            "textAlign": "center",
            "padding": "20px"
        })
    
    df = pd.DataFrame(data)
    greek_columns = ["Delta", "Theta", "Vega", "Rho", "Gamma", "Vanna", "Vomma", "Charm", "Color", "Speed", "Ultima", "Zomma"]
    price_columns = ["Mid-Point Price", "BSM Price", "Monte Carlo Price"]
    
    greek_agg = {col: 0 for col in greek_columns}
    price_agg = {col: 0 for col in price_columns}
    
    r = 0.05
    opt_type = "c"
    
    for i in selected_rows:
        row = df.iloc[i]
        trade_type = row.get("Trade Type", "Buy")
        greek_multiplier = 1 if trade_type == "Buy" else -1
        price_multiplier = -1 if trade_type == "Buy" else 1
        
        try:
            original_iv = float(row.get("Implied Vol.", 0))
        except (ValueError, TypeError):
            original_iv = 0.0
        # Get underlying price; override if price_override is provided.
        try:
            S = float(price_override) if price_override is not None else float(row.get("Underlying Price"))
        except (ValueError, TypeError):
            S = 0.0
        try:
            K = float(row.get("strike"))
        except (ValueError, TypeError):
            K = 0.0
        try:
            T = float(row.get("Days to Expiry", 0)) / 365
        except (ValueError, TypeError):
            T = 0.0
        
        if iv_adjustment is not None:
            try:
                iv_delta = float(iv_adjustment) / 100  # Convert percentage to decimal
            except (ValueError, TypeError):
                iv_delta = 0.0
        else:
            iv_delta = 0.0
        new_iv = original_iv + iv_delta
        
        temp = row.copy()
        temp["Implied Vol."] = new_iv
        
        new_delta  = delta(temp, S, r=r, option_type=opt_type) or 0
        new_theta  = theta(temp, S, r=r, option_type=opt_type) or 0
        new_vega   = vega(temp, S) or 0
        new_rho    = rho(temp, S, r=r, option_type=opt_type) or 0
        new_gamma  = gamma(temp, S, r=r) or 0
        new_vanna  = vanna(temp, S, r=r) or 0
        new_vomma  = vomma(temp, S, r=r) or 0
        new_charm  = charm(temp, S, r=r, option_type=opt_type) or 0
        new_color  = color(temp, S, r=r) or 0
        new_speed  = speed(temp, S, r=r) or 0
        new_ultima = ultima(temp, S, r=r) or 0
        new_zomma  = zomma(temp, S, r=r) or 0
        
        greek_agg["Delta"] += greek_multiplier * new_delta
        greek_agg["Theta"] += greek_multiplier * new_theta
        greek_agg["Vega"]  += greek_multiplier * new_vega
        greek_agg["Rho"]   += greek_multiplier * new_rho
        greek_agg["Gamma"] += greek_multiplier * new_gamma
        greek_agg["Vanna"] += greek_multiplier * new_vanna
        greek_agg["Vomma"] += greek_multiplier * new_vomma
        greek_agg["Charm"] += greek_multiplier * new_charm
        greek_agg["Color"] += greek_multiplier * new_color
        greek_agg["Speed"] += greek_multiplier * new_speed
        greek_agg["Ultima"] += greek_multiplier * new_ultima
        greek_agg["Zomma"] += greek_multiplier * new_zomma
        
        new_bsm = bsm_price_calc(S, K, T, new_iv, opt_type)
        new_mc  = monte_carlo_price_calc(S, K, T, new_iv, opt_type)
        try:
            mid_point = float(row.get("Mid-Point Price", 0))
        except (ValueError, TypeError):
            mid_point = 0.0
        
        price_agg["Mid-Point Price"] += price_multiplier * mid_point
        price_agg["BSM Price"] += price_multiplier * new_bsm
        price_agg["Monte Carlo Price"] += price_multiplier * new_mc
        
        df.at[i, "New BSM Price"] = new_bsm
        df.at[i, "New Monte Carlo Price"] = new_mc
        df.at[i, "New Delta"] = new_delta
        df.at[i, "New Theta"] = new_theta
        df.at[i, "New Vega"] = new_vega
        df.at[i, "New Rho"] = new_rho
        df.at[i, "New Gamma"] = new_gamma
    
    # Define styles for headers and alternating rows with gridlines
    header_style = {
        "fontWeight": "bold", 
        "padding": "8px", 
        "fontFamily": "Helvetica Neue, Arial, sans-serif", 
        "textAlign": "center", 
        "backgroundColor": "#f2f2f2", 
        "border": "1px solid #ddd"
    }
    cell_style_even = {
        "padding": "8px", 
        "fontFamily": "Helvetica Neue, Arial, sans-serif", 
        "textAlign": "center", 
        "border": "1px solid #ddd", 
        "backgroundColor": "#f9f9f9"
    }
    cell_style_odd = {
        "padding": "8px", 
        "fontFamily": "Helvetica Neue, Arial, sans-serif", 
        "textAlign": "center", 
        "border": "1px solid #ddd", 
        "backgroundColor": "white"
    }
    
    # Build the Greeks table using adjusted aggregated values
    greek_table_rows = [html.Tr([
        html.Td("Metric", style=header_style),
        html.Td("Net Value (Adjusted)", style=header_style)
    ])]
    for idx, (col, val) in enumerate(greek_agg.items()):
        row_style = cell_style_even if idx % 2 == 0 else cell_style_odd
        greek_table_rows.append(html.Tr([
            html.Td(col, style=row_style),
            html.Td(f"{val:.4f}", style=row_style)
        ]))
    greek_table = html.Table(greek_table_rows, style={
        "margin": "auto", 
        "border": "1px solid #ddd", 
        "borderCollapse": "collapse", 
        "width": "50%"
    })
    
    # Build the Price table (using recalculated prices)
    price_table_rows = [html.Tr([
        html.Td("Price Metric", style=header_style),
        html.Td("Net Premium/Cost (Adjusted)", style=header_style)
    ])]
    for idx, (col, val) in enumerate(price_agg.items()):
        row_style = cell_style_even if idx % 2 == 0 else cell_style_odd
        price_table_rows.append(html.Tr([
            html.Td(col, style=row_style),
            html.Td(f"{val:.4f}", style=row_style)
        ]))
    price_table = html.Table(price_table_rows, style={
        "margin": "auto", 
        "border": "1px solid #ddd", 
        "borderCollapse": "collapse", 
        "width": "50%"
    })
    
    # Build the Selected Contracts table with details (including new recalculated prices)
    contract_table_rows = [html.Tr([
        html.Td("Strike", style=header_style),
        html.Td("Mid-Point Price", style=header_style),
        html.Td("Original BSM Price", style=header_style),
        html.Td("New BSM Price", style=header_style),
        html.Td("Original Monte Carlo Price", style=header_style),
        html.Td("New Monte Carlo Price", style=header_style),
        html.Td("Days to Expiry", style=header_style),
        html.Td("Trade Type", style=header_style)
    ])]
    for idx, i in enumerate(selected_rows):
        row = df.iloc[i]
        row_style = cell_style_even if idx % 2 == 0 else cell_style_odd
        contract_table_rows.append(html.Tr([
            html.Td(row.get("strike", ""), style=row_style),
            html.Td(row.get("Mid-Point Price", ""), style=row_style),
            html.Td(row.get("BSM Price", ""), style=row_style),
            html.Td(row.get("New BSM Price", ""), style=row_style),
            html.Td(row.get("Monte Carlo Price", ""), style=row_style),
            html.Td(row.get("New Monte Carlo Price", ""), style=row_style),
            html.Td(row.get("Days to Expiry", ""), style=row_style),
            html.Td(row.get("Trade Type", ""), style=row_style)
        ]))
    contract_table = html.Table(contract_table_rows, style={
        "margin": "auto", 
        "border": "1px solid #ddd", 
        "borderCollapse": "collapse", 
        "width": "80%"
    })
    
    return html.Div([
        html.H3("Net Greeks (Adjusted by IV)", style={
            "fontFamily": "Helvetica Neue, Arial, sans-serif",
            "fontSize": "24px",
            "fontWeight": "bold",
            "textAlign": "center",
            "marginTop": "20px"
        }),
        greek_table,
        html.Br(),
        html.H3("Net Premium / Cost (Adjusted by IV)", style={
            "fontFamily": "Helvetica Neue, Arial, sans-serif",
            "fontSize": "24px",
            "fontWeight": "bold",
            "textAlign": "center",
            "marginTop": "20px"
        }),
        price_table,
        html.Br(),
        html.H3("Selected Contracts", style={
            "fontFamily": "Helvetica Neue, Arial, sans-serif",
            "fontSize": "24px",
            "fontWeight": "bold",
            "textAlign": "center",
            "marginTop": "20px"
        }),
        contract_table
    ])

# Add new callback for payoff diagram
@app.callback(
    [Output('payoff_chart', 'figure'),
     Output('payoff_metrics', 'children')],
    [Input('options_table', 'data'),
     Input('options_table', 'selected_rows'),
     Input('opt_type', 'value')]
)
def update_payoff_diagram(data, selected_rows, opt_type):
    if not data or not selected_rows or len(selected_rows) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No options selected",
            template="plotly_dark",
            paper_bgcolor='#282a36',
            plot_bgcolor='#282a36'
        )
        return empty_fig, "Please select options to view payoff diagram"
    
    selected_options = [data[i] for i in selected_rows]
    for option in selected_options:
        option['option_type'] = opt_type
    
    price_points, payoff, breakeven_points, max_profit, max_loss = calculate_payoff_points(selected_options)
    
    if price_points is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Unable to calculate payoff",
            template="plotly_dark",
            paper_bgcolor='#282a36',
            plot_bgcolor='#282a36'
        )
        return empty_fig, "Error calculating payoff"
    
    # Create the payoff diagram
    fig = go.Figure()
    
    # Add the profit/loss line
    fig.add_trace(go.Scatter(
        x=price_points,
        y=payoff,
        mode='lines',
        name='P/L',
        line=dict(color='#50fa7b', width=2)
    ))
    
    # Add breakeven points
    if breakeven_points:
        fig.add_trace(go.Scatter(
            x=breakeven_points,
            y=[0] * len(breakeven_points),
            mode='markers',
            name='Breakeven',
            marker=dict(color='#bd93f9', size=10, symbol='diamond')
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#6272a4")
    
    # Update layout
    fig.update_layout(
        title="Option Position Payoff Diagram",
        xaxis_title="Underlying Price",
        yaxis_title="Profit/Loss",
        template="plotly_dark",
        paper_bgcolor='#282a36',
        plot_bgcolor='#282a36',
        hovermode='x unified',
        showlegend=True
    )
    
    # Create metrics display
    metrics_div = html.Div([
        html.Div([
            html.Span("Breakeven Points: ", style={'fontWeight': 'bold'}),
            html.Span(f"{', '.join(map(str, breakeven_points))}" if breakeven_points else "N/A")
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Span("Max Profit: ", style={'fontWeight': 'bold'}),
            html.Span(f"{'∞' if max_profit == float('inf') else f'${max_profit:.2f}' if max_profit != float('-inf') else 'Unlimited'}")
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Span("Max Loss: ", style={'fontWeight': 'bold'}),
            html.Span(f"{'∞' if max_loss == float('-inf') else f'${max_loss:.2f}' if max_loss != float('inf') else 'Unlimited'}")
        ])
    ])
    
    return fig, metrics_div

# Add download callbacks
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-csv", "n_clicks"),
    State('options_table', 'data'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, data):
    if not n_clicks or not data:
        raise PreventUpdate
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "options_chain.csv", index=False)

@app.callback(
    Output("download-dataframe-excel", "data"),
    Input("btn-excel", "n_clicks"),
    State('options_table', 'data'),
    prevent_initial_call=True,
)
def download_excel(n_clicks, data):
    if not n_clicks or not data:
        raise PreventUpdate
    df = pd.DataFrame(data)
    
    # Create an Excel writer object
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Options Chain', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Options Chain']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#44475a',
            'font_color': '#f8f8f2',
            'border': 1
        })
        
        # Format the header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Auto-adjust columns' width
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_len = max(
                series.astype(str).map(len).max(),
                len(str(series.name))
            ) + 1
            worksheet.set_column(idx, idx, max_len)
    
    return dcc.send_bytes(output.getvalue(), "options_chain.xlsx")

# Add callbacks for Iron Condor builder
@app.callback(
    Output('ic_expiry', 'options'),
    [Input('ic_ticker', 'value')]
)
def update_ic_expiry(ticker):
    if not ticker:
        return []
    try:
        ticker_obj = yf.Ticker(ticker)
        expiry_dates = ticker_obj.options
        return [{'label': date, 'value': date} for date in expiry_dates]
    except:
        return []

@app.callback(
    [Output('ic_short_call', 'options'),
     Output('ic_long_call', 'options'),
     Output('ic_short_put', 'options'),
     Output('ic_long_put', 'options')],
    [Input('ic_ticker', 'value'),
     Input('ic_expiry', 'value')]
)
def update_ic_strikes(ticker, expiry):
    if not ticker or not expiry:
        return [], [], [], []
    try:
        ticker_obj = yf.Ticker(ticker)
        current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
        option_chain = ticker_obj.option_chain(expiry)
        
        # Get strikes around current price
        calls = option_chain.calls
        puts = option_chain.puts
        
        # Filter strikes to reasonable range
        call_strikes = calls[calls['strike'] >= current_price * 0.95]['strike'].tolist()
        put_strikes = puts[puts['strike'] <= current_price * 1.05]['strike'].tolist()
        
        # Create options for dropdowns
        call_options = [{'label': f'${strike:.2f}', 'value': strike} for strike in call_strikes]
        put_options = [{'label': f'${strike:.2f}', 'value': strike} for strike in put_strikes]
        
        return call_options, call_options, put_options, put_options
    except:
        return [], [], [], []

@app.callback(
    [Output('ic_analysis', 'children'),
     Output('ic_payoff_chart', 'figure')],
    [Input('ic_ticker', 'value'),
     Input('ic_expiry', 'value'),
     Input('ic_short_call', 'value'),
     Input('ic_long_call', 'value'),
     Input('ic_short_put', 'value'),
     Input('ic_long_put', 'value')]
)
def update_ic_analysis(ticker, expiry, short_call, long_call, short_put, long_put):
    if not all([ticker, expiry, short_call, long_call, short_put, long_put]):
        return "Please select all strikes to analyze the strategy.", go.Figure()
    
    try:
        ticker_obj = yf.Ticker(ticker)
        current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
        option_chain = ticker_obj.option_chain(expiry)
        
        # Get option prices
        short_call_price = option_chain.calls[option_chain.calls['strike'] == short_call]['lastPrice'].iloc[0]
        long_call_price = option_chain.calls[option_chain.calls['strike'] == long_call]['lastPrice'].iloc[0]
        short_put_price = option_chain.puts[option_chain.puts['strike'] == short_put]['lastPrice'].iloc[0]
        long_put_price = option_chain.puts[option_chain.puts['strike'] == long_put]['lastPrice'].iloc[0]
        
        # Calculate strategy metrics
        max_profit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
        max_loss = (long_call - short_call) - max_profit
        
        # Calculate breakeven points
        upper_breakeven = short_call + max_profit
        lower_breakeven = short_put - max_profit
        
        # Create payoff diagram
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
        payoff = np.zeros_like(price_range)
        
        for i, price in enumerate(price_range):
            # Short call payoff
            payoff[i] += short_call_price - max(price - short_call, 0)
            # Long call payoff
            payoff[i] += -long_call_price + max(price - long_call, 0)
            # Short put payoff
            payoff[i] += short_put_price - max(short_put - price, 0)
            # Long put payoff
            payoff[i] += -long_put_price + max(long_put - price, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_range,
            y=payoff,
            mode='lines',
            name='P/L',
            line=dict(color='#50fa7b', width=2)
        ))
        
        # Add breakeven points
        fig.add_trace(go.Scatter(
            x=[upper_breakeven, lower_breakeven],
            y=[0, 0],
            mode='markers',
            name='Breakeven',
            marker=dict(color='#bd93f9', size=10, symbol='diamond')
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#6272a4")
        
        fig.update_layout(
            title="Strategy Payoff Diagram",
            xaxis_title="Underlying Price",
            yaxis_title="Profit/Loss",
            template="plotly_dark",
            paper_bgcolor='#282a36',
            plot_bgcolor='#282a36',
            hovermode='x unified',
            showlegend=True
        )
        
        # Create analysis text
        analysis = html.Div([
            html.Div([
                html.Span("Maximum Profit: ", style={'fontWeight': 'bold'}),
                html.Span(f"${max_profit:.2f}")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span("Maximum Loss: ", style={'fontWeight': 'bold'}),
                html.Span(f"${max_loss:.2f}")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span("Upper Breakeven: ", style={'fontWeight': 'bold'}),
                html.Span(f"${upper_breakeven:.2f}")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span("Lower Breakeven: ", style={'fontWeight': 'bold'}),
                html.Span(f"${lower_breakeven:.2f}")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span("Profit Zone: ", style={'fontWeight': 'bold'}),
                html.Span(f"${lower_breakeven:.2f} to ${upper_breakeven:.2f}")
            ])
        ])
        
        return analysis, fig
        
    except Exception as e:
        return f"Error analyzing strategy: {str(e)}", go.Figure()

# ---------------------------
# Run the App
# ---------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
