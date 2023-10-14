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
		// parameters in linear term.
		-0.0676945966455097,
		0.4440724481952758,
		0.6303579668166768,
		-0.3131943974231358,
		0.20629609481418254,
		// latent vector in interaction term.
		0.39247746871954314,
		-0.5476150804831619,
		-0.03396464094835039,
		// latent vector in interaction term.
		0.16615141245739373,
		-0.7802055920361937,
		-0.275942049244708,
		// latent vector in interaction term.
		-0.0585761116186686,
		0.9858117104860424,
		-0.5643249592026998,
		// latent vector in interaction term.
		0.47754886028288457,
		0.7397362300575543,
		-0.7930005163459948,
		// latent vector in interaction term.
		0.25143606218376835,
		0.5136688711322772,
		-0.5030497609372591,
	}

	tests := []struct {
		name string
		want []float64
	}{
		{
			name: "`At` method returns target the parameter value specified by the index",
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

func TestFMParamsToLatentVecElementIndex(t *testing.T) {
	randoms := []float64{
		// parameters in linear term.
		-0.0676945966455097,
		0.4440724481952758,
		0.6303579668166768,
		-0.3131943974231358,
		0.20629609481418254,
		// latent vector in interaction term.
		0.39247746871954314,
		-0.5476150804831619,
		-0.03396464094835039,
		// latent vector in interaction term.
		0.16615141245739373,
		-0.7802055920361937,
		-0.275942049244708,
		// latent vector in interaction term.
		-0.0585761116186686,
		0.9858117104860424,
		-0.5643249592026998,
		// latent vector in interaction term.
		0.47754886028288457,
		0.7397362300575543,
		-0.7930005163459948,
		// latent vector in interaction term.
		0.25143606218376835,
		0.5136688711322772,
		-0.5030497609372591,
	}

	paramIndexToLatentVectorEleIndex := map[int]int{
		5: 0,
		6: 1,
		7: 2,

		8:  0,
		9:  1,
		10: 2,

		11: 0,
		12: 1,
		13: 2,

		14: 0,
		15: 1,
		16: 2,

		17: 0,
		18: 1,
		19: 2,
	}

	type args struct {
		paramIndices []int
	}
	tests := []struct {
		name string
		args args
		want map[int]int
	}{
		{
			name: "ToLatentVecElementIndex method maps the parameter index into the index in target latent vector element",
			args: args{
				paramIndices: []int{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
			},
			want: paramIndexToLatentVectorEleIndex,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fp := NewFMParams(3, 5, WithRandomFunc(newDummyRandom(randoms).randomFunc))
			for _, paramIndex := range tt.args.paramIndices {
				if got := fp.ToLatentVecElementIndex(paramIndex); got != tt.want[paramIndex] {
					t.Errorf("ToLatentVecElementIndex() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestFMParamsToLatentVecIndex(t *testing.T) {
	randoms := []float64{
		// parameters in linear term.
		-0.0676945966455097,
		0.4440724481952758,
		0.6303579668166768,
		-0.3131943974231358,
		0.20629609481418254,
		// latent vector in interaction term.
		0.39247746871954314,
		-0.5476150804831619,
		-0.03396464094835039,
		// latent vector in interaction term.
		0.16615141245739373,
		-0.7802055920361937,
		-0.275942049244708,
		// latent vector in interaction term.
		-0.0585761116186686,
		0.9858117104860424,
		-0.5643249592026998,
		// latent vector in interaction term.
		0.47754886028288457,
		0.7397362300575543,
		-0.7930005163459948,
		// latent vector in interaction term.
		0.25143606218376835,
		0.5136688711322772,
		-0.5030497609372591,
	}

	paramIndexToLatentVectorIndex := map[int]int{
		5: 0,
		6: 0,
		7: 0,

		8:  1,
		9:  1,
		10: 1,

		11: 2,
		12: 2,
		13: 2,

		14: 3,
		15: 3,
		16: 3,

		17: 4,
		18: 4,
		19: 4,
	}

	type args struct {
		paramIndices []int
	}
	tests := []struct {
		name string
		args args
		want map[int]int
	}{
		{
			name: "ToLatentVecIndex method maps the parameter index into the index in target latent vector",
			args: args{
				paramIndices: []int{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
			},
			want: paramIndexToLatentVectorIndex,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fp := NewFMParams(3, 5, WithRandomFunc(newDummyRandom(randoms).randomFunc))
			for _, paramIndex := range tt.args.paramIndices {
				if got := fp.ToLatentVecIndex(paramIndex); got != tt.want[paramIndex] {
					t.Errorf("ToLatentVecIndex() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestFMParamsUpdate(t *testing.T) {
	type fields struct {
		useBias     bool
		bias        float64
		Linear      FMLinear
		Interact    FMInteraction
		totalLen    int
		linearLen   int
		interactLen int
		latestDim   int
		randomFunc  func(float64, float64) float64
	}
	type args struct {
		paramIndex int
		value      float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fp := FMParams{
				useBias:     tt.fields.useBias,
				bias:        tt.fields.bias,
				Linear:      tt.fields.Linear,
				Interact:    tt.fields.Interact,
				totalLen:    tt.fields.totalLen,
				linearLen:   tt.fields.linearLen,
				interactLen: tt.fields.interactLen,
				latestDim:   tt.fields.latestDim,
				randomFunc:  tt.fields.randomFunc,
			}
			fp.Update(tt.args.paramIndex, tt.args.value)
		})
	}
}
