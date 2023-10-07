package gofm

import (
	"testing"
)

func (d dummyRandom) randomFunc(min, max float64) float64 {
	i := *d.index
	*d.index++
	return d.randoms[i]
}

type dummyRandom struct {
	index   *int
	randoms []float64
}

func newDummyRandom(randoms []float64) dummyRandom {
	initIndex := 0
	return dummyRandom{
		index:   &initIndex,
		randoms: randoms,
	}
}

func TestFMParamsAt(t *testing.T) {

	randoms := []float64{
		-0.0676945966455097,
		0.4440724481952758,
		0.6303579668166768,
		-0.3131943974231358,
		0.20629609481418254,
		0.39247746871954314,
		-0.5476150804831619,
		-0.03396464094835039,
		0.16615141245739373,
		-0.7802055920361937,
		-0.275942049244708,
		-0.0585761116186686,
		0.9858117104860424,
		-0.5643249592026998,
		0.47754886028288457,
		0.7397362300575543,
		-0.7930005163459948,
		0.25143606218376835,
		0.5136688711322772,
		-0.5030497609372591,
	}

	tests := []struct {
		name string
		want []float64
	}{
		{
			name: "get target value from FMParams",
			want: randoms,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fp := NewFMParams(3, 5, WithRandomFunc(newDummyRandom(randoms).randomFunc))
			for i := 0; i < fp.totalLen; i++ {
				got := fp.At(i)
				if got != tt.want[i] {
					t.Errorf("At() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}
