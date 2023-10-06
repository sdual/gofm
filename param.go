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
		useBias:   false,
		Linear:    NewFMLinear(featureDim),
		Interact:  NewFMInteraction(featureDim, latentDim),
		latestDim: latentDim,
	}
	params.totalLen = params.totalNumParams()
	params.linearLen = featureDim
	params.interactLen = params.totalLen - featureDim

	for _, opt := range opts {
		opt(&params)
	}
	return params
}

func UseBaias(useBias bool) func(params *FMParams) {
	return func(p *FMParams) {
		p.useBias = useBias
	}
}

func NewFMLinear(dim int) FMLinear {
	params := make([]float64, dim)
	for i := 0; i < dim; i++ {
		params = append(params, RandomFloat64(-1.0, 1.0))
	}
	return FMLinear{
		params: params,
	}
}

func NewFMInteraction(numInteraction int, latentDim int) FMInteraction {
	vecs := make([]FMLatentVec, numInteraction)
	for i := 0; i < numInteraction; i++ {
		elements := make([]float64, latentDim)
		for j := 0; j < latentDim; j++ {
			elements = append(elements, RandomFloat64(-1.0, 1.0))
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
		interactionIndex := i - (fp.linearLen - 1)
		vecIndex := interactionIndex/fp.latestDim - 1
		latentVec := fp.Interact.latentVecs[vecIndex]

		// index in latent vector
		latentVecIndex := interactionIndex - (vecIndex+1)*fp.latestDim
		return latentVec.elements[latentVecIndex]
	}
}

func (fp FMParams) totalNumParams() int {
	linearLen := len(fp.Linear.params)
	interactLen := 0
	for _, vec := range fp.Interact.latentVecs {
		interactLen += len(vec.elements)
	}
	return linearLen
}
