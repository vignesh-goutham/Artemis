package stocks

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/shopspring/decimal"
	"github.com/vignesh-goutham/artemis/pkg/alpaca"
	"github.com/vignesh-goutham/artemis/pkg/dynamo"
	"github.com/vignesh-goutham/artemis/pkg/types"
)

// Handler processes trades and manages the bot flow
func Handler(ctx context.Context) (map[string]interface{}, error) {
	// Check if market is open
	clock, err := alpaca.GetTradingClient().GetClock()
	if err != nil {
		return nil, err
	}
	if !clock.IsOpen {
		msg := fmt.Sprintf("Market is closed. Next open: %s", clock.NextOpen.Format(time.RFC3339))
		log.Println(msg)
		return map[string]interface{}{
			"statusCode": 200,
			"body":       msg,
		}, nil
	}

	// Get active trades from DynamoDB
	trades, err := dynamo.GetAllTrades(ctx)
	if err != nil {
		return nil, err
	}

	if len(trades) == 0 {
		msg := "No active trades to process"
		log.Println(msg)
		return map[string]interface{}{
			"statusCode": 200,
			"body":       msg,
		}, nil
	}

	// Get account value
	accountValue, err := alpaca.GetAccountValue()
	if err != nil {
		return nil, err
	}
	accountValueDecimal := decimal.NewFromFloat(accountValue)

	// Get unique tickers for processing
	tickers := make([]string, 0, len(trades))
	tickerMap := make(map[string]bool)
	for _, trade := range trades {
		if !tickerMap[trade.Ticker] {
			tickers = append(tickers, trade.Ticker)
			tickerMap[trade.Ticker] = true
		}
	}

	// Get historical data for all tickers
	startDate := time.Now().AddDate(0, 0, -100)
	historicalData, err := alpaca.GetStockDataBatch(tickers, startDate, time.Now())
	if err != nil {
		return nil, err
	}

	// Get current prices
	currentPrices, err := alpaca.GetCurrentPrices(tickers)
	if err != nil {
		return nil, err
	}

	// Process each trade
	var updates []types.Trade
	var deletes []string
	var skippedTrades []string
	var skippedReasons []string
	for _, trade := range trades {
		updatedTrade, shouldUpdate, err := processTrade(trade, historicalData, accountValueDecimal, currentPrices)
		if err != nil {
			log.Printf("Error processing trade for %s: %v", trade.Ticker, err)
			skippedTrades = append(skippedTrades, trade.Ticker)
			skippedReasons = append(skippedReasons, err.Error())
			continue
		}

		if shouldUpdate {
			if updatedTrade.Status == "COMPLETED" {
				deletes = append(deletes, updatedTrade.Ticker)
			} else {
				updates = append(updates, updatedTrade)
			}
		}
	}

	// Batch write updates and deletes
	if len(updates) > 0 || len(deletes) > 0 {
		if err := dynamo.BatchWriteTrades(ctx, updates, deletes); err != nil {
			log.Printf("Error batch writing trades: %v", err)
			return nil, err
		}
	}

	// Prepare response
	msg := fmt.Sprintf("Trading strategy executed successfully. Processed %d trades, skipped %d trades", len(trades), len(skippedTrades))
	if len(skippedTrades) > 0 {
		msg += "\nSkipped trades:"
		for i, ticker := range skippedTrades {
			msg += fmt.Sprintf("\n  - %s: %s", ticker, skippedReasons[i])
		}
	}

	return map[string]interface{}{
		"statusCode": 200,
		"body":       msg,
	}, nil
}
