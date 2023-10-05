package gofm

import (
	"fmt"
	"math"
)

const epsilon = 1e-7

func BinaryCrossEntropy(targets []float64, preds []float64) float64 {
	if len(targets) != len(preds) {
		panic(fmt.Sprintf("targets must have the same length of preds. targets: %d, preds: %d", len(targets), len(preds)))
	}
	if len(targets) == 0 {
		return 0.0
	}

	value := 0.0
	for i, target := range targets {
		value += -target*math.Log(preds[i]+epsilon) - (1.0-target)*math.Log(1.0-preds[i]+epsilon)
	}
	return value / float64(len(targets))
}

func DiffBinaryCrossEntropy() float64 {
	return 0.0
}
