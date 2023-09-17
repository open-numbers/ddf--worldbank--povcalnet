# How to use the etl script

## Dependencies
This repo uses [waf](https://waf.io/) as its build system. So you should have the `waf` command in your PATH.

I am choosing waf because:
- waf script are pure python.
- waf processes files only when necessary by recording "what has
  changed". The etl for this dataset consists of some independent
  steps and it takes a long time to run the entire process. So it's
  good to have this feature.

And before you run waf commands, it's recommended that you create and activate a Python virtual environment.

## Run the build

``` shell
$ waf configure update_source update_source_gapminder buildall
```

The above command will create shapes from povcalnet data in
`etl/build` folder, population_500plus.pkl for total population by
bracket and population_percentage_500plus.pkl for population
percentage by bracket.

These files then can be used in other scripts to create datapoints we needed:

1. run combine_billionaires.py to generate the bridged shapes with povcalnet and billionaires data (bridged_shapes.pkl)
2. run entities.py to generate entities files
3. run datapoints.py and datapoint_bridge.py to generate the datapoint files for income mountain
4. run decile_income.py to create decile/centile income data

## TODO items

- [ ] use waf to install venv and requirements
- [ ] use waf to handle all processes (combine billionaries and income mountain datapoints etc)
- [ ] update datapoints.py and datapoint_bridge.py, might be good to combine them into one file
