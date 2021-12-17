import datetime
import calendar

def to_fractional_year(d: datetime.datetime) -> float:
    """
    Converts a date stamp to a fractional year (i.e. number like `1939.781`)
    """
    is_leap = calendar.isleap(d.year)
    t = d.timetuple()
    day_of_year = t.tm_yday
    day_time = (60 * 60 * t.tm_hour + 60 * t.tm_min + t.tm_sec) / (24 * 60 * 60)

    days_in_year = 366 if is_leap else 365

    return d.year + ((day_of_year - 1 + day_time) / days_in_year)

def fractional_to_date(fractional):
    eps = 0.0001
    year = int(fractional)
    is_leap = calendar.isleap(year)
    
    modulus = fractional % 1

    days_in_year = 366 if is_leap else 365

    day_of_year =  int( days_in_year * modulus + eps )

    d = datetime.datetime(year, 1,1) + datetime.timedelta(days = day_of_year )

    return d

dates = (datetime.datetime(1825,10,30),
        datetime.datetime(1825,10,31),
        datetime.datetime(1900,1,1),
	datetime.datetime(1900,12,1),
        datetime.datetime(1900,12,31),
        datetime.datetime(1930,2,28),
        datetime.datetime(1932,2,29),
	)

for split in 'train', 'dev-0':
    with open(f'../{split}/in.tsv') as f_in, open(f'../{split}/expected.tsv') as f_exp, open(f'./{split}_huggingface_format.csv', 'w') as f_hf:
        f_hf.write('year_cont\tyear\tmonth\tday\tweekday\tday_of_year\ttext\n')
        for line_in,line_exp in zip(f_in,f_exp):
            year_cont = float(line_exp.rstrip())
            date = fractional_to_date(year_cont)
            year = date.year
            month = date.month
            day = date.day
            weekday = date.weekday()
            day_of_year = date.timetuple().tm_yday

            #f_hf.write(line_exp.rstrip() + '\t' + line_in)
            f_hf.write(f'{year_cont}\t{year}\t{month}\t{day}\t{weekday}\t{day_of_year}\t{line_in}')

for split in ('test-A',):
    with open(f'../{split}/in.tsv') as f_in,  open(f'./{split}_huggingface_format.csv', 'w') as f_hf:
        f_hf.write('year_cont\tyear\tmonth\tday\tweekday\tday_of_year\ttext\n')
        for line_in in f_in:
            f_hf.write(f'0\t0\t0\t0\t0\t0\t{line_in}')
