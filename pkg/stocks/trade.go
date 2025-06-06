package stocks

import (
	"fmt"
	"log"
	"time"

	alpacatrade "github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
	"github.com/shopspring/decimal"
	"github.com/vignesh-goutham/artemis/pkg/alpaca"
	"github.com/vignesh-goutham/artemis/pkg/types"
)

func processTrade(trade types.Trade, historicalData map[string]alpaca.StockData, accountValue decimal.Decimal,
	currentPrices map[string]decimal.Decimal) (types.Trade, bool, error) {
	currentDate := time.Now()
	hist := historicalData[trade.Ticker]

	log.Printf("Processing trade for %s: Status=%s, TotalShares=%s, BuyPrice=$%s",
		trade.Ticker, trade.Status, trade.TotalShares, trade.BuyPrice)

	currentPrice, exists := currentPrices[trade.Ticker]
	if !exists {
		return trade, false, fmt.Errorf("no valid price available for ticker %s", trade.Ticker)
	}

	// Validate current price
	if currentPrice.IsZero() {
		return trade, false, fmt.Errorf("invalid price ($%s) for ticker %s", currentPrice, trade.Ticker)
	}

	log.Printf("Current price for %s: $%s", trade.Ticker, currentPrice)

	// Compare dates using YYYY-MM-DD format
	currentDateStr := currentDate.Format("2006-01-02")
	sellDateStr := trade.SellDate.Format("2006-01-02")
	buyDateStr := trade.BuyDate.Format("2006-01-02")

	log.Printf("Date comparison for %s - Current: %s, Buy: %s, Sell: %s",
		trade.Ticker, currentDateStr, buyDateStr, sellDateStr)

	if currentDateStr >= sellDateStr {
		log.Printf("Sell date reached for %s (SellDate: %s)", trade.Ticker, sellDateStr)
		updatedTrade, sold, err := executeSell(trade, currentPrice)
		if err != nil {
			return trade, false, fmt.Errorf("failed to execute sell for %s: %v", trade.Ticker, err)
		}

		if sold {
			log.Printf("Successfully sold %s: TotalShares=%s, Price=$%s, P/L=$%s",
				trade.Ticker, updatedTrade.TotalShares, updatedTrade.SellPrice, updatedTrade.ProfitLoss)
			return updatedTrade, true, nil
		}

	} else if currentDateStr >= buyDateStr && trade.Status != "ACTIVE" {
		log.Printf("Buy date reached for %s (BuyDate: %s)", trade.Ticker, buyDateStr)
		updatedTrade, _, err := executeBuy(trade, currentPrice, accountValue)
		if err != nil {
			return trade, false, fmt.Errorf("failed to execute buy for %s: %v", trade.Ticker, err)
		}

		log.Printf("Successfully bought %s: TotalShares=%s, Price=$%s",
			trade.Ticker, updatedTrade.TotalShares, updatedTrade.BuyPrice)
		trade = updatedTrade
	} else if trade.Status == "ACTIVE" {
		if len(trade.DipBuys) < 2 && checkDipBuyConditions(hist, trade.BuyPrice, &trade.LastDipBuyDate, currentPrice) {
			log.Printf("Dip buy conditions met for %s: CurrentPrice=$%s, BuyPrice=$%s, DipBuys=%d",
				trade.Ticker, currentPrice, trade.BuyPrice, len(trade.DipBuys))
			updatedTrade, _, err := executeDipBuy(trade, currentPrice)
			if err != nil {
				return trade, false, fmt.Errorf("failed to execute dip buy for %s: %v", trade.Ticker, err)
			}

			log.Printf("Successfully executed dip buy for %s: Additional Shares=%s, Price=$%s",
				trade.Ticker, updatedTrade.TotalShares.Sub(trade.TotalShares), currentPrice)
			trade = updatedTrade
		} else {
			log.Printf("No dip buy for %s: Conditions not met or max dip buys reached", trade.Ticker)
		}
	}

	return trade, false, nil
}

// executeBuy executes a buy order for a trade
func executeBuy(trade types.Trade, currentPrice, accountValue decimal.Decimal) (types.Trade, bool, error) {
	// Calculate investment amount based on allocation percentage
	investmentAmount := accountValue.Mul(trade.AllocationPercentage.Div(decimal.NewFromInt(100)))
	shares := investmentAmount.Div(currentPrice)

	// Execute the trade through Alpaca
	_, err := alpaca.ExecuteTrade(trade.Ticker, alpacatrade.Buy, shares.InexactFloat64(), currentPrice.InexactFloat64())
	if err != nil {
		return trade, false, fmt.Errorf("failed to execute buy order: %v", err)
	}

	// Update trade with order details
	trade.Status = "ACTIVE"
	trade.BuyPrice = currentPrice
	trade.InitialShares = shares
	trade.TotalShares = shares
	trade.InvestmentAmount = investmentAmount
	trade.TotalInvestment = investmentAmount

	return trade, true, nil
}

// executeSell executes a sell order for a trade
func executeSell(trade types.Trade, currentPrice decimal.Decimal) (types.Trade, bool, error) {
	// Execute the sell trade through Alpaca
	_, err := alpaca.ExecuteTrade(trade.Ticker, alpacatrade.Sell, trade.TotalShares.InexactFloat64(), currentPrice.InexactFloat64())
	if err != nil {
		return trade, false, fmt.Errorf("failed to execute sell order: %v", err)
	}

	// Calculate profit/loss
	trade.SellPrice = currentPrice
	trade.ProfitLoss = trade.TotalShares.Mul(currentPrice.Sub(trade.BuyPrice))
	trade.ProfitLossPercentage = trade.ProfitLoss.Div(trade.TotalInvestment).Mul(decimal.NewFromInt(100))
	trade.Status = "COMPLETED"

	return trade, true, nil
}

// executeDipBuy executes an additional buy during a price dip
func executeDipBuy(trade types.Trade, currentPrice decimal.Decimal) (types.Trade, bool, error) {
	// Calculate additional investment (same as initial)
	additionalInvestment := trade.InvestmentAmount
	additionalShares := additionalInvestment.Div(currentPrice)

	// Execute the dip buy trade through Alpaca
	_, err := alpaca.ExecuteTrade(trade.Ticker, alpacatrade.Buy, additionalShares.InexactFloat64(), currentPrice.InexactFloat64())
	if err != nil {
		return trade, false, fmt.Errorf("failed to execute dip buy order: %v", err)
	}

	// Update position
	trade.AdditionalShares = trade.AdditionalShares.Add(additionalShares)
	trade.AdditionalInvestment = trade.AdditionalInvestment.Add(additionalInvestment)
	trade.TotalShares = trade.TotalShares.Add(additionalShares)
	trade.TotalInvestment = trade.TotalInvestment.Add(additionalInvestment)

	// Record dip buy
	trade.DipBuys = append(trade.DipBuys, types.DipBuy{
		Date:       time.Now(),
		Price:      currentPrice,
		Shares:     additionalShares,
		Investment: additionalInvestment,
	})
	trade.LastDipBuyDate = time.Now()

	return trade, true, nil
}

// checkDipBuyConditions checks if conditions are met for a dip buy
func checkDipBuyConditions(hist alpaca.StockData, buyPrice decimal.Decimal, lastDipBuyDate *time.Time, currentPrice decimal.Decimal) bool {
	if len(hist.Dates) == 0 {
		return false
	}

	// Get current SMA20
	currentSMA := decimal.NewFromFloat(hist.SMA20[len(hist.SMA20)-1])

	// Find highest price since buy
	highestPrice := decimal.Zero
	for i, date := range hist.Dates {
		if date.After(*lastDipBuyDate) {
			high := decimal.NewFromFloat(hist.High[i])
			if high.GreaterThan(highestPrice) {
				highestPrice = high
			}
		}
	}

	// Calculate price change from high
	priceChangeFromHigh := highestPrice.Sub(currentPrice).Div(highestPrice)

	// Check conditions:
	// 1. Price has dropped at least 10% from recent high
	// 2. Current price is below SMA20
	// 3. SMA20 is above our initial buy price
	// 4. Current price is below our initial buy price
	// 5. At least 5 days since last dip buy
	return priceChangeFromHigh.GreaterThanOrEqual(decimal.NewFromFloat(0.10)) &&
		currentPrice.LessThan(currentSMA) &&
		currentSMA.GreaterThan(buyPrice) &&
		currentPrice.LessThan(buyPrice) &&
		(lastDipBuyDate.IsZero() || time.Since(*lastDipBuyDate).Hours() >= 24*5)
}
