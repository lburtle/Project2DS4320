# From Weather Records to Climate Signals: Forecasting What the Data Is Telling Us About Virginia's Future

## Hook
Every single day, weather stations across Virginia record temperature highs and lows, 
rainfall totals, wind speeds, and solar radiation. Year after year, that data accumulates 
into an enormous archive of atmospheric history. But most people only look at weather one 
day at a time. The real story is not what happened yesterday, but is what fifteen years 
of daily observations, taken together, reveal about where the climate is heading. The 
question is not whether the climate is changing. It is whether we can measure exactly 
how, where, and how fast.

## Problem Statement
Traditional climate reporting gives you global averages: "The Earth warmed by 1.1°C 
since pre-industrial times." That number is accurate, but it is also abstract. It tells 
a resident of Bristol, Virginia nothing about whether their winters are getting shorter, 
whether summer heat extremes are intensifying, or whether precipitation patterns in the 
Shenandoah Valley are shifting. A global average is forced to collapse an enormous range 
of local variation into a single number, and in doing so, it loses the information that 
actually matters for local planning, agriculture, and infrastructure. Without 
location-specific trend analysis and credible forward forecasts, communities cannot make 
informed decisions about a changing climate.

## Solution Description
This project proposes moving from global averages to location-specific signals by 
training models directly on fifteen years of daily weather observations across eight 
geographically diverse locations in Virginia. Rather than producing a single point 
prediction, the pipeline incorporates a Temporal Fusion Transformer that generates 
probabilistic forecasts with prediction intervals, communicating not just where 
temperatures are heading but how much uncertainty surrounds that projection. In addition 
to the deep learning approach, a Random Forest model trained on lag and rolling features 
provides an interpretable baseline, with feature importances revealing which past 
observations most strongly drive the forecast. By combining long-range historical data, 
multiple model classes, and probabilistic output, the result is a forecast that reflects 
the complexity of local climate rather than smoothing over it.
