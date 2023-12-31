package gofm

import (
	"math/rand"
	"time"
)

func RandomFloat64(min, max float64) float64 {
	rand.Seed(time.Now().UnixNano())
	return rand.Float64()*(max-min) + min
}
