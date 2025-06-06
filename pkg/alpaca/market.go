package alpaca

import (
	"fmt"
	"time"

	"github.com/alpacahq/alpaca-trade-api-go/v3/marketdata"
	"github.com/shopspring/decimal"
)

// GetStockDataBatch retrieves historical data for multiple tickers
func GetStockDataBatch(tickers []string, startDate, endDate time.Time) (map[string]StockData, error) {
	result := make(map[string]StockData)

	bars, err := GetDataClient().GetMultiBars(tickers, marketdata.GetBarsRequest{
		Start:     startDate,
		End:       endDate,
		TimeFrame: marketdata.OneDay,
		Feed:      marketdata.IEX,
	})
	if err != nil {
		return nil, fmt.Errorf("error getting bars: %v", err)
	}

	for ticker, tickerBars := range bars {
		close := make([]float64, len(tickerBars))
		high := make([]float64, len(tickerBars))
		dates := make([]time.Time, len(tickerBars))

		for i, bar := range tickerBars {
			close[i] = bar.Close
			high[i] = bar.High
			dates[i] = bar.Timestamp
		}

		// Calculate SMA20
		sma20 := make([]float64, len(close))
		for i := 19; i < len(close); i++ {
			sum := 0.0
			for j := 0; j < 20; j++ {
				sum += close[i-j]
			}
			sma20[i] = sum / 20
		}

		result[ticker] = StockData{
			Close: close,
			High:  high,
			SMA20: sma20,
			Dates: dates,
		}
	}

	return result, nil
}

// GetCurrentPrices retrieves current prices for multiple tickers
func GetCurrentPrices(tickers []string) (map[string]decimal.Decimal, error) {
	quotes, err := GetDataClient().GetLatestQuotes(tickers, marketdata.GetLatestQuoteRequest{
		Feed: marketdata.IEX,
	})
	if err != nil {
		return nil, fmt.Errorf("error getting latest quotes: %v", err)
	}

	prices := make(map[string]decimal.Decimal)
	for ticker, quote := range quotes {
		if quote.AskPrice <= 0 || quote.BidPrice <= 0 {
			continue
		}
		prices[ticker] = decimal.NewFromFloat(quote.AskPrice)
	}
	return prices, nil
}
