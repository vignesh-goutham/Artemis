package alpaca

import (
	"time"
)

// StockData represents processed stock data with technical indicators
type StockData struct {
	Close []float64
	High  []float64
	SMA20 []float64
	Dates []time.Time
}
