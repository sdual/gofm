package gofm

import (
	"fmt"
	"math"
)

const epsilon = 1e-7

func TotalBinaryCrossEntropy(targets []float64, preds []float64) float64 {
	if len(targets) != len(preds) {
		panic(fmt.Sprintf("targets must have the same length of preds. targets: %d, preds: %d", len(targets), len(preds)))
	}
	if len(targets) == 0 {
		return 0.0
	}

	value := 0.0
	for index, target := range targets {
		value += BinaryCrossEntropy(target, preds[index])
	}
	return value / float64(len(targets))
}

func BinaryCrossEntropy(target float64, pred float64) float64 {
	return -target*math.Log(pred+epsilon) - (1.0-target)*math.Log(1.0-pred+epsilon)
}

func DiffBinaryCrossEntropy() float64 {
	return 0.0
}
