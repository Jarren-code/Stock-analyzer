from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import akshare as ak
import yfinance as yf  #requests might be blocked
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
import plotly.utils
import json
import os


app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")


# ==================== MODEL CLASS ====================
class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ==================== LOAD MODEL & SCALER ====================
seq_length = 60

if not os.path.exists('models/model.pth'):
    print("Error: models/model.pth not found!")
    exit(1)

if not os.path.exists('models/scaler.pkl'):
    print("Error: models/scaler.pkl not found!")
    exit(1)

model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
model.load_state_dict(torch.load('models/model.pth', map_location=device))
model.eval()

scaler = joblib.load('models/scaler.pkl')

print("✅ Model & scaler loaded successfully!")

#translate chinese data into english data
def fetch_stock_data_akshare(ticker, period="1y"):
    """
    Fetches stock data using AKShare as fallback for Chinese stocks.
    """
    try:
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y%m%d")
        if period == "1y":
            start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y%m%d")
        elif period == "6mo":
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        else:
            start_date = (datetime.now() - timedelta(days=1 * 365)).strftime("%Y%m%d")

        # Try US stocks first
        try:
            df = ak.stock_us_hist(symbol=ticker, period="daily",
                                  start_date=start_date, end_date=end_date, adjust="")
            if not df.empty:
                df = df.rename(columns={
                    '日期': 'Date',
                    '开盘': 'Open',
                    '最高': 'High',
                    '最低': 'Low',
                    '收盘': 'Close',
                    '成交量': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                return df
        except:
            pass

        # Try A-shares (Chinese stocks)
        try:
            df = ak.stock_zh_a_hist(symbol=ticker, period="daily",
                                    start_date=start_date, end_date=end_date, adjust="")
            if not df.empty:
                df = df.rename(columns={
                    '日期': 'Date',
                    '开盘': 'Open',
                    '最高': 'High',
                    '最低': 'Low',
                    '收盘': 'Close',
                    '成交量': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                return df
        except:
            pass

        return pd.DataFrame()
    except Exception as e:
        print(f"AKShare error for {ticker}: {e}")
        return pd.DataFrame()

# ==================== PREDICTION FUNCTION ====================
def predict_price(ticker):
    try:
        # Try yfinance first (for US stocks)
        try:
            df = yf.download(ticker, period='1y', progress=False, timeout=10)
            if not df.empty:
                print(f"✅ yfinance worked for {ticker}")
            else:
                raise Exception("yfinance returned empty")
        except:
            # Fallback to AKShare
            print(f"🔄 Trying AKShare for {ticker}")
            df = fetch_stock_data_akshare(ticker, period="1y")
            if df.empty:
                return None, None

        data = df['Close'].values.reshape(-1, 1)
        scaled = scaler.transform(data)

        if len(scaled) < seq_length:
            return None, None

        X_input = scaled[-seq_length:].reshape(1, seq_length, 1)
        X_tensor = torch.FloatTensor(X_input).to(device)

        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()

        pred_price = scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
        current_price = float(df['Close'].iloc[-1])

        return current_price, pred_price
    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None, None


# ==================== CHART FUNCTION ====================
def get_chart(ticker):
    try:
        # Try yfinance first
        try:
            df = yf.download(ticker, period='3mo', progress=False, timeout=10)
            if df.empty:
                raise Exception("yfinance returned empty")
        except:
            # Fallback to AKShare
            print(f"🔄 AKShare fallback for chart: {ticker}")
            df = fetch_stock_data_akshare(ticker, period="3mo")
            if df.empty:
                return None

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])

        fig.update_layout(
            title=f'{ticker} Stock Price',
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400
        )

        return fig.to_dict()
    except Exception as e:
        print(f"Chart error for {ticker}: {e}")
        return None


# ==================== SCORING FUNCTIONS ====================
def safe_float(value, default=0):
    """Safely convert any value to float, handling pandas Series and None."""
    if value is None:
        return float(default)
    if isinstance(value, (pd.Series, pd.DataFrame)):
        try:
            return float(value.iloc[0])
        except:
            return float(default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def calculate_moat_score(ticker):
    """Durable Competitive Advantage Score (0-100)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # ✅ Use safe_float to avoid pandas Series issues
        roic = safe_float(info.get('returnOnCapital'), 0)
        gross_margin = safe_float(info.get('grossMargins'), 0)
        debt_to_equity = safe_float(info.get('debtToEquity'), 1.0)
        fcf = safe_float(info.get('freeCashflow'), 0)
        earnings_growth = safe_float(info.get('earningsGrowth'), 0)

        # ROIC
        if roic > 0.15:
            roic_score = 100
        elif roic > 0.10:
            roic_score = 70
        elif roic > 0.05:
            roic_score = 40
        else:
            roic_score = 10

        # Gross Margin
        if gross_margin > 0.40:
            margin_score = 100
        elif gross_margin > 0.25:
            margin_score = 60
        else:
            margin_score = 30

        # Debt-to-Equity
        if debt_to_equity < 0.5:
            debt_score = 100
        elif debt_to_equity < 1.0:
            debt_score = 70
        elif debt_to_equity < 1.5:
            debt_score = 40
        else:
            debt_score = 20

        # Free Cash Flow
        fcf_score = 100 if fcf > 0 else 30

        # Earnings Growth
        if earnings_growth > 0.10:
            growth_score = 100
        elif earnings_growth > 0:
            growth_score = 70
        else:
            growth_score = 40

        return min(100, max(0, (
                roic_score * 0.30 + margin_score * 0.20 + growth_score * 0.25 + debt_score * 0.15 + fcf_score * 0.10
        )))
    except Exception as e:
        print(f"Moat score error for {ticker}: {e}")
        return 50


def calculate_mos_score(ticker):
    """Margin of Safety Score (0-100)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # ✅ Use safe_float
        pe = safe_float(info.get('trailingPE'), 0)
        pb = safe_float(info.get('priceToBook'), 0)
        peg = safe_float(info.get('pegRatio'), 0)

        # P/E Score
        sector_avg_pe = 20
        if pe > 0 and pe < sector_avg_pe * 0.8:
            pe_score = 100
        elif pe > 0 and pe < sector_avg_pe:
            pe_score = 70
        elif pe > 0 and pe < sector_avg_pe * 1.5:
            pe_score = 40
        else:
            pe_score = 10

        # P/B Score
        if pb > 0 and pb < 1.5:
            pb_score = 100
        elif pb > 0 and pb < 3:
            pb_score = 60
        else:
            pb_score = 20

        # PEG Score
        if peg > 0 and peg < 1:
            peg_score = 100
        elif peg > 0 and peg < 2:
            peg_score = 60
        else:
            peg_score = 20

        # DCF Score (placeholder)
        dcf_score = 50

        return min(100, max(0, (pe_score * 0.30 + pb_score * 0.20 + peg_score * 0.20 + dcf_score * 0.30)))
    except Exception as e:
        print(f"MOS score error for {ticker}: {e}")
        return 50


def calculate_sentiment_score(ticker):
    """Sentiment Score (0-100)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        recommendation = info.get('recommendationKey', 'hold')
        if recommendation is None:
            recommendation = 'hold'

        rec_map = {'strong_buy': 100, 'buy': 75, 'hold': 50, 'sell': 25, 'strong_sell': 0}
        analyst_score = rec_map.get(recommendation, 50)

        # Placeholder for news sentiment
        news_score = 50

        return min(100, max(0, (analyst_score * 0.50 + news_score * 0.50)))
    except Exception as e:
        print(f"Sentiment error for {ticker}: {e}")
        return 50


def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return None

        return {
            'symbol': ticker,
            'name': str(info.get('longName', 'N/A')),
            'sector': str(info.get('sector', 'N/A')),
            'industry': str(info.get('industry', 'N/A')),
            'country': str(info.get('country', 'N/A')),
            'market_cap': f"${safe_float(info.get('marketCap'), 0):,.0f}" if safe_float(info.get('marketCap'),
                                                                                        0) > 0 else 'N/A',
            'pe_ratio': round(safe_float(info.get('trailingPE'), 0), 2) if safe_float(info.get('trailingPE'),
                                                                                      0) > 0 else 'N/A',
            'dividend_yield': safe_float(info.get('dividendYield'), 0),
            'website': str(info.get('website', 'N/A')),
            'description': str(info.get('longBusinessSummary', 'No description available.'))
        }
    except Exception as e:
        print(f"Info error for {ticker}: {e}")
        return None


# ==================== MAIN ANALYSIS ENDPOINT ====================
@app.route('/analyze/<ticker>')
def analyze(ticker):
    """Full analysis endpoint."""
    try:
        # 1. Get price prediction
        current_price, predicted_price = predict_price(ticker)
        if current_price is None:
            return jsonify({'error': 'Stock not found or insufficient data'}), 404

        change_pct = ((predicted_price / current_price) - 1) * 100
        direction = 'up' if change_pct > 0 else 'down'

        # 2. Get company info
        company_info = get_company_info(ticker)

        # 3. Calculate scores
        moat = calculate_moat_score(ticker)
        mos = calculate_mos_score(ticker)
        sentiment = calculate_sentiment_score(ticker)

        # 4. Long-term framework
        long_score = (moat * 0.55) + (mos * 0.30) + (sentiment * 0.15)
        long_verdict = 'BUY' if long_score >= 70 else 'HOLD' if long_score >= 50 else 'SELL'

        # 5. Short-term framework
        short_score = (moat * 0.20) + (mos * 0.45) + (sentiment * 0.35)
        short_verdict = 'BUY' if short_score >= 70 else 'HOLD' if short_score >= 50 else 'SELL'

        # 6. Get chart
        chart_data = get_chart(ticker)

        return jsonify({
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'change_pct': round(change_pct, 2),
            'direction': direction,
            'company': company_info,
            'long_term': {
                'score': round(long_score, 1),
                'verdict': long_verdict,
                'components': {
                    'Moat (55%)': round(moat, 1),
                    'MOS (30%)': round(mos, 1),
                    'Sentiment (15%)': round(sentiment, 1)
                }
            },
            'short_term': {
                'score': round(short_score, 1),
                'verdict': short_verdict,
                'components': {
                    'Moat (20%)': round(moat, 1),
                    'MOS (45%)': round(mos, 1),
                    'Sentiment (35%)': round(sentiment, 1)
                }
            },
            'chart': chart_data
        })
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== OTHER ROUTES ====================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'AAPL').upper()

    try:
        current, pred = predict_price(ticker)
        if current is None or pred is None:
            return jsonify({'error': 'Stock not found or insufficient data'}), 404

        change = ((pred / current) - 1) * 100

        return jsonify({
            'ticker': ticker,
            'current_price': round(current, 2),
            'predicted_price': round(pred, 2),
            'change_pct': round(change, 2),
            'direction': 'up' if change > 0 else 'down'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chart/<ticker>')
def chart(ticker):
    try:
        chart_data = get_chart(ticker)
        if chart_data is None:
            return jsonify({'error': 'No chart data found'}), 404
        return jsonify({'chart': chart_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/info/<ticker>')
def info(ticker):
    try:
        company_info = get_company_info(ticker)
        if company_info is None:
            return jsonify({'error': 'Stock not found'}), 404
        return jsonify(company_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== HEALTH CHECK ====================
@app.route('/')
def home():
    return jsonify({
        'message': '📈 Stock Predictor API is running!',
        'endpoints': {
            '/analyze/<ticker>': 'Full stock analysis',
            '/predict': 'Price prediction (POST)',
            '/chart/<ticker>': 'Chart data',
            '/info/<ticker>': 'Company info'
        },
        'status': 'healthy'
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)