SELECT DISTINCT pqc_name, pqc_sso
FROM pqc_case_assignments 
WHERE pqc_name IN (
    'Nick Vigorito', 
    'Angela Young', 
    'Jenn Parsons', 
    'Patricia Gomes',
    'Cicily Raj',
    'Savithry Kandregula',
    'Ayesha Ahmed',
    'Pavani Shelly',
    'Rupa Talla',
    'Rupa T'
)
ORDER BY pqc_name;
