package types

import (
	"time"

	"github.com/shopspring/decimal"
)

// Trade represents a trading strategy entry
type Trade struct {
	// Core fields
	Ticker               string          `dynamodbav:"ticker"`
	BuyDate              time.Time       `dynamodbav:"buy_date"`
	SellDate             time.Time       `dynamodbav:"sell_date"`
	Status               string          `dynamodbav:"status"`
	AllocationPercentage decimal.Decimal `dynamodbav:"allocation_percentage"`

	// Price and position information
	BuyPrice             decimal.Decimal `dynamodbav:"buy_price,omitempty"`
	SellPrice            decimal.Decimal `dynamodbav:"sell_price,omitempty"`
	InitialShares        decimal.Decimal `dynamodbav:"initial_shares,omitempty"`
	AdditionalShares     decimal.Decimal `dynamodbav:"additional_shares,omitempty"`
	TotalShares          decimal.Decimal `dynamodbav:"total_shares,omitempty"`
	InvestmentAmount     decimal.Decimal `dynamodbav:"investment_amount,omitempty"`
	AdditionalInvestment decimal.Decimal `dynamodbav:"additional_investment,omitempty"`
	TotalInvestment      decimal.Decimal `dynamodbav:"total_investment,omitempty"`
	ProfitLoss           decimal.Decimal `dynamodbav:"profit_loss,omitempty"`
	ProfitLossPercentage decimal.Decimal `dynamodbav:"profit_loss_percentage,omitempty"`

	// Dip buying information
	LastDipBuyDate time.Time `dynamodbav:"last_dip_buy_date,omitempty"`
	DipBuys        []DipBuy  `dynamodbav:"dip_buys,omitempty"`
}

// DipBuy represents an additional buy during a price dip
type DipBuy struct {
	Date       time.Time       `dynamodbav:"date"`
	Price      decimal.Decimal `dynamodbav:"price"`
	Shares     decimal.Decimal `dynamodbav:"shares"`
	Investment decimal.Decimal `dynamodbav:"investment"`
}
