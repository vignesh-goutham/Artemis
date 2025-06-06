package alpaca

import (
	"log"
	"os"

	"github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
	"github.com/alpacahq/alpaca-trade-api-go/v3/marketdata"
)

var (
	tradingClient *alpaca.Client
	dataClient    *marketdata.Client
)

// Initialize sets up the Alpaca clients
func Initialize() error {
	tradingClient = alpaca.NewClient(alpaca.ClientOpts{
		APIKey:    os.Getenv("ALPACA_API_KEY"),
		APISecret: os.Getenv("ALPACA_SECRET_KEY"),
		BaseURL:   "https://paper-api.alpaca.markets",
	})

	dataClient = marketdata.NewClient(marketdata.ClientOpts{
		APIKey:    os.Getenv("ALPACA_API_KEY"),
		APISecret: os.Getenv("ALPACA_SECRET_KEY"),
	})

	return nil
}

// GetTradingClient returns the Alpaca trading client instance
func GetTradingClient() *alpaca.Client {
	if tradingClient == nil {
		log.Fatal("Alpaca trading client not initialized")
	}
	return tradingClient
}

// GetDataClient returns the Alpaca market data client instance
func GetDataClient() *marketdata.Client {
	if dataClient == nil {
		log.Fatal("Alpaca market data client not initialized")
	}
	return dataClient
}
