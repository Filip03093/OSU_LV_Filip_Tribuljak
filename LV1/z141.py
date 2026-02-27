def total_euro(hours, wage):
    return hours * wage

hours = int(input('Radni sati: '))
wage = float(input('eura/h: '))

print(f"Ukupno: {total_euro(hours, wage)} eura")