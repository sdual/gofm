package gofm

import "testing"

func TestTotalBinaryCrossEntropy(t *testing.T) {
	type args struct {
		targets []float64
		preds   []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "TotalBinaryCrossEntropy function takes target and prediction values, and calculate cross entropy",
			args: args{
				targets: []float64{1.0, 0.0, 1.0},
				preds:   []float64{0.4, 0.5, 0.9},
			},
			want: 0.5715992889936243,
		},
		{
			name: "if TotalBinaryCrossEntropy function takes empty slices, 0.0 is returned",
			args: args{
				targets: []float64{},
				preds:   []float64{},
			},
			want: 0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := TotalBinaryCrossEntropy(tt.args.targets, tt.args.preds); got != tt.want {
				t.Errorf("TotalBinaryCrossEntropy() = %v, want %v", got, tt.want)
			}
		})
	}
}
