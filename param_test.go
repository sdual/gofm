package gofm

import (
	"testing"
)

type dummyRandom struct {
	index *int
}

func (d dummyRandom) randomFunc(min, max float64) float64 {
	i := *d.index
	*d.index++
	return float64(i)
}

func TestFMParamsAt(t *testing.T) {
	tests := []struct {
		name string
		want float64
	}{
		{
			name: "get target value from FMParams",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startNum := 0
			fp := NewFMParams(3, 5, WithRandomFunc(dummyRandom{index: &startNum}.randomFunc))

			for i := 0; i < fp.totalLen; i++ {
				got := fp.At(i)
				if got != float64(i) {
					t.Errorf("At() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}
