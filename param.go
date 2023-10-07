package gofm

import (
	"gonum.org/v1/gonum/mat"
)

const (
	paramInitMax = 1.0
	paramInitMin = -1.0
)

type (
	// FMHyperparams contains hyperparameters for Factorization machine.
	FMHyperparams struct {
		latentDim int64
	}

	// FMParams contains model parameters for Factorization machine.
	FMParams struct {
		useBias     bool    // Bias is not supported now.
		bias        float64 // Bias is not supported now.
		Linear      FMLinear
		Interact    FMInteraction
		totalLen    int
		linearLen   int
		interactLen int
		latestDim   int
		randomFunc  func(float64, float64) float64
	}

	// FMLinear contains parameters used in linear terms.
	FMLinear struct {
		params []float64
	}

	// FMInteraction contains parameters used in interaction terms.
	FMInteraction struct {
		latentVecs []FMLatentVec
	}

	// FMLatentVec contains latent vector used in interaction terms.
	FMLatentVec struct {
		vector mat.Vector
	}
)

func NewFMParams(latentDim int, featureDim int, opts ...func(params *FMParams)) FMParams {
	params := FMParams{
		useBias:     false,
		latestDim:   latentDim,
		totalLen:    featureDim + featureDim*latentDim,
		linearLen:   featureDim,
		interactLen: featureDim * latentDim,
	}

	for _, opt := range opts {
		opt(&params)
	}

	if params.randomFunc == nil {
		params.randomFunc = RandomFloat64
	}

	params.Linear = NewFMLinear(featureDim, params.randomFunc)
	params.Interact = NewFMInteraction(featureDim, latentDim, params.randomFunc)
	return params
}

func WithBaiasUsed(useBias bool) func(params *FMParams) {
	return func(p *FMParams) {
		p.useBias = useBias
	}
}

func WithRandomFunc(f func(float64, float64) float64) func(params *FMParams) {
	return func(p *FMParams) {
		p.randomFunc = f
	}
}

func NewFMLinear(dim int, randomFunc func(float64, float64) float64) FMLinear {
	params := make([]float64, 0, dim)
	for i := 0; i < dim; i++ {
		params = append(params, randomFunc(paramInitMin, paramInitMax))
	}
	return FMLinear{
		params: params,
	}
}

func NewFMInteraction(numInteraction int, latentDim int, randomFunc func(float64, float64) float64) FMInteraction {
	vecs := make([]FMLatentVec, 0, numInteraction)
	for i := 0; i < numInteraction; i++ {
		elements := make([]float64, 0, latentDim)
		for j := 0; j < latentDim; j++ {
			elements = append(elements, randomFunc(paramInitMin, paramInitMax))
		}
		vecs = append(vecs, FMLatentVec{vector: mat.NewVecDense(latentDim, elements)})
	}

	return FMInteraction{
		latentVecs: vecs,
	}
}

func (fp FMParams) Update() {

}

// At returns the parameter value specified by the index.
// Unified indices are assigned to the entirety of parameters in the linear and interaction terms.
func (fp FMParams) At(i int) float64 {
	if i < fp.linearLen {
		return fp.Linear.params[i]
	} else {
		// index in interaction terms
		interactionIndex := i - fp.linearLen
		vecIndex := interactionIndex / fp.latestDim
		latentVec := fp.Interact.latentVecs[vecIndex]

		// index in latent vector
		latentVecIndex := interactionIndex - vecIndex*fp.latestDim
		return latentVec.vector.AtVec(latentVecIndex)
	}
}

// LatentVecFromFeatureIndex returns the latent vector which corresponds to the feature index.
// In interaction terms Σ_{i,j} <v_i, v_j> x_i x_j, the feature index i correspond to latent vector v_i.
func (fp FMParams) LatentVecFromFeatureIndex(i int) mat.Vector {
	return fp.Interact.latentVecs[i].vector
}

// ToLatentVecElementIndex maps the unified param index into the index of element in the latent vector.
func (fp FMParams) ToLatentVecElementIndex(i int) int {
	if i < fp.linearLen {
		panic("index must be in interaction terms")
	}

	interactParamIndex := i - fp.linearLen
	indexInLatentVector := interactParamIndex % fp.latestDim
	return indexInLatentVector
}

// ToLatentVecIndex maps the unified param index into the index of the latent vector slice.
// This method returns the index of latent vector in list of latent vectors.
// The latent vector contains the parameter specified by the index.
func (fp FMParams) ToLatentVecIndex(i int) int {
	if i < fp.linearLen {
		panic("index must be in interaction terms")
	}
	interactParamIndex := i - fp.linearLen
	latentVecIndex := interactParamIndex / fp.latestDim
	return latentVecIndex
}

func (fp FMParams) IsLinear(index int) bool {
	return index < fp.linearLen
}
