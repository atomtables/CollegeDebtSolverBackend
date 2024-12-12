import base64
import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from matplotlib.patheffects import Normal, SimpleLineShadow

mpl.use('Agg')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

polygon_api_key = "YzQk9Qzpr5jIGOUhPH3p5YQywSjizNEp"
client = RESTClient(polygon_api_key)

stockTicker = 'AAPL'


@app.get("/stock/{symbol}")
def get_stock_data(symbol: str):
    # daily bars
    dataRequest = client.get_aggs(ticker=symbol,
                                  multiplier=30,
                                  timespan='day',
                                  from_='2022-09-01',
                                  to='2100-01-01')
    if dataRequest == []:
        return {"error": "invalid symbol"}
    print(dataRequest)
    priceData = pd.DataFrame(dataRequest)
    return (priceData)


@app.get("/stock/{symbol}/generalInfo")
def get_stock_graph(symbol: str):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    hi = hist['Close']
    if hi.iloc[0] < hi.iloc[-1]:
        print("Stock price increased")
        color = "green"
    else:
        print("Stock price decreased")
        color = "red"
    fig, ax = plt.subplots()
    ax.figure.set_figwidth(16)
    ax.figure.set_figheight(9)
    plt.rcParams["font.family"] = "Roboto"
    hi.plot(ax=ax,
            color=color,
            linewidth=3,
            path_effects=[
                SimpleLineShadow(shadow_color=color, linewidth=5),
                Normal()
            ])
    ax.legend(["Stock Price"])
    plt.savefig("stock.png", dpi=175)
    my_string_IO_bytes = io.BytesIO()
    plt.savefig(my_string_IO_bytes, format='jpg')
    my_string_IO_bytes.seek(0)
    return {
        "netIncome": stock.income_stmt.iloc[:, 0].get('Net Income', None),
        "netDebt": stock.balance_sheet.iloc[:, 0].get('Net Debt', None),
        "sharesIssued": stock.balance_sheet.iloc[:,
                                                 0].get('Shares Issued', None),
        "graph": str(base64.b64encode(my_string_IO_bytes.read()).decode()),
        "graphData": {
            "start": hi.iloc[0],
            "startDate": hi.index[0],
            "end": hi.iloc[-1],
            "endDate": hi.index[-1],
            "mean": hi.mean(),
            "median": hi.median(),
            "mode": hi.mode()[0],
        },
        "info": stock.info,
        "news": stock.news
    }
