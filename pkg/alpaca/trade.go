package alpaca

import (
	"fmt"
	"log"
	"math"

	"github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
	"github.com/shopspring/decimal"
)

// ExecuteTrade executes a trade with the given parameters
func ExecuteTrade(ticker string, side alpaca.Side, qty float64, currentPrice float64) (*alpaca.Order, error) {
	limitPrice := decimal.NewFromFloat(math.Round(currentPrice*100) / 100)
	quantity := decimal.NewFromFloat(qty)

	log.Printf("Executing %s order for %s: Quantity=%.2f, Price=$%.2f",
		side, ticker, qty, currentPrice)

	order, err := GetTradingClient().PlaceOrder(alpaca.PlaceOrderRequest{
		Symbol:      ticker,
		Qty:         &quantity,
		Side:        side,
		Type:        alpaca.Limit,
		TimeInForce: alpaca.Day,
		LimitPrice:  &limitPrice,
	})
	if err != nil {
		log.Printf("Error placing %s order for %s: %v", side, ticker, err)
		return nil, fmt.Errorf("error placing order: %v", err)
	}

	log.Printf("Successfully placed %s order for %s: OrderID=%s, Status=%s",
		side, ticker, order.ID, order.Status)
	return order, nil
}

// GetAccountValue retrieves the current account value
func GetAccountValue() (float64, error) {
	account, err := GetTradingClient().GetAccount()
	if err != nil {
		return 0, fmt.Errorf("error getting account: %v", err)
	}
	equity, _ := account.Equity.Float64()
	return equity, nil
}
