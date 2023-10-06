package gofm

type FMLeaner struct {
	gd          GradientDescent
	hyperParams FMHyperParams
	Params      FMParams
}

func NewFMLeaner(gd GradientDescent) FMLeaner {
	return FMLeaner{
		gd: gd,
	}
}
