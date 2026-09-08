## Briefs on `synth_02.wav`

Translation model `claude-haiku-4-5`. 8 briefs, 8 translation calls costing $0.0394 to record and nothing to replay; 1 needed a JSON repair.

### Translation against controller

The same recorded translations driving different controllers, so the `translation` column is identical by construction and any difference in the other two belongs to the controller alone.

| controller | translation | execution | collateral | passed | controller cost |
|---|---|---|---|---|---|
| `agent-scaffold` | 94% | 75% | 88% | 4/8 | $0.0000 |
| `agent` | 94% | 56% | 85% | 2/8 | $0.1117 |


### Controller `agent-scaffold`

Nothing here is judged by a model. Each brief carries the regions it must move and the direction, written down in advance, and all three columns are arithmetic. `translation` is whether the target named the region correctly, `execution` whether the render actually moved it, and `collateral` whether the families the brief said to leave alone stayed inside tolerance. An expectation fails if anything in its region moved the wrong way.

| brief | translation | execution | collateral | renders | cost |
|---|---|---|---|---|---|
| Make it brighter and more open up top. | 100% | 100% | 67% | 2 | $0.0000 |
| Warmer and rounder in the low mids, please. | 100% | 100% | 100% | 2 | $0.0000 |
| More space and width, but keep the low end ... | 50% | 50% | 33% | 3 | $0.0000 |
| Pull the stereo image in, it is too wide an... | 100% | 50% | 100% | 7 | $0.0000 |
| Take the harshness out of the upper mids. | 100% | 100% | 100% | 1 | $0.0000 |
| Get it up to streaming level without squash... | 100% | 100% | 100% | 1 | $0.0000 |
| It needs more punch and impact. | 100% | 0% | 100% | 1 | $0.0000 |
| Scoop the boxy mids out of it. | 100% | 100% | 100% | 2 | $0.0000 |

4 of 8 briefs satisfied all three. Means: translation 94%, execution 75%, collateral 88%. Controller cost $0.0000.


### Controller `agent`

Nothing here is judged by a model. Each brief carries the regions it must move and the direction, written down in advance, and all three columns are arithmetic. `translation` is whether the target named the region correctly, `execution` whether the render actually moved it, and `collateral` whether the families the brief said to leave alone stayed inside tolerance. An expectation fails if anything in its region moved the wrong way.

| brief | translation | execution | collateral | renders | cost |
|---|---|---|---|---|---|
| Make it brighter and more open up top. | 100% | 100% | 33% | 8 | $0.0378 |
| Warmer and rounder in the low mids, please. | 100% | 0% | 100% | 3 | $0.0152 |
| More space and width, but keep the low end ... | 50% | 50% | 67% | 3 | $0.0170 |
| Pull the stereo image in, it is too wide an... | 100% | 0% | 100% | 5 | $0.0291 |
| Take the harshness out of the upper mids. | 100% | 100% | 100% | 1 | $0.0034 |
| Get it up to streaming level without squash... | 100% | 100% | 100% | 1 | $0.0051 |
| It needs more punch and impact. | 100% | 0% | 100% | 0 | $0.0000 |
| Scoop the boxy mids out of it. | 100% | 100% | 82% | 1 | $0.0040 |

2 of 8 briefs satisfied all three. Means: translation 94%, execution 56%, collateral 85%. Controller cost $0.1117.

