package gofm

type (
	// FMHyperParams contains hyper parameters for Factorization machine.
	FMHyperParams struct {
		latentDim int64
	}

	// FMParams contains model parameters for Factorization machine.
	FMParams struct {
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
		elements []float64
	}
)

func NewFMParams(latentDim int, featureDim int, opts ...func(params *FMParams)) FMParams {
	params := FMParams{
		useBias:    false,
		latestDim:  latentDim,
		randomFunc: RandomFloat64,
	}
	params.linearLen = featureDim
	params.totalLen = params.totalNumParams(featureDim)
	params.interactLen = params.totalLen - featureDim

	for _, opt := range opts {
		opt(&params)
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
		params = append(params, randomFunc(-1.0, 1.0))
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
			elements = append(elements, randomFunc(-1.0, 1.0))
		}
		vecs = append(vecs, FMLatentVec{elements: elements})
	}

	return FMInteraction{
		latentVecs: vecs,
	}
}

func (fp FMParams) Update() {

}

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
		return latentVec.elements[latentVecIndex]
	}
}

func (fp FMParams) totalNumParams(featureDim int) int {
	totalLen := featureDim
	for _, vec := range fp.Interact.latentVecs {
		totalLen += len(vec.elements)
	}
	return totalLen
}
