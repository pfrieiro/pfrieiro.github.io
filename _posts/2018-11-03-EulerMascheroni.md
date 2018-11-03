---
layout: post
title: "Paralelización en julia"
categories: julia mathematics parallelization
math: true
---

# Introducción

En este post se van a explorar las capacidades nativas de paralelización en
julia.

# La constante de Euler-Mascheroni

Como parte de los ejercicios de la asignatura de Cálculo Paralelo que cursé en
el [máster](http://www.m2i.es/) se pedía realizar un pequeño programa en C
utilizando OpenMPI para calcular la constante de Euler-Mascheroni (hasta 20
dígitos).

Esta constante se obtiene como el límite de la diferencia entre la serie
armónica y el logaritmo natural:

$$
\gamma = \lim_{n\to\infty}\left( \sum_{k=1}^n \frac{1}{k} - \log n\right)
$$

El valor de ésta constante puede consultarse en la página para la secuencia
[A001620 en la OEIS](https://oeis.org/A001620), y por ejemplo los primeros 50
dígitos son:

$$
\gamma \approx 0.57721\;56649\;01532\;86060\;65120\;90082\;40243\;10421\;59335\;93992...
$$

El cálculo de ésta constante es interesante puesto que la fórmula mostrada en la
definición tiene una convergencia muy lenta, del orden de $\mathcal{O}(n^-1)$,
donde $n$ es el número de términos de la serie armónica.
Esto quiere decir que el error cometido tomando $n=1000$ será de aproximadamente
$10^-3$ y por ello que podemos esperar solo unos tres decimales de precisión.
Por tanto, el número de decimales correctos es del orden $\mathcal{O}(\log n)$
por lo que obtener 20 dígitos exige una gran cantidad de operaciones.

Veamos esto en julia.

```julia
"""
Función para calcular la serie armónica.
"""
function harmonic(n::Int64)
    x = 0
    for i in n:-1:1 # se comienza sumando desde el término más pequeño
        x +=1/i     # para evitar errores de truncadura.
    end
    x
end

"""
Función para calcular la constante de Euler-Mascheroni.
"""
function compute_γ(n::Int64)
    harmonic(n) - log(n)
end
```

Se calcula para $n=10^6$:

```julia
compute_γ(1_000_000)
```
```bash
0.57721(61649014986)
```

Por supuesto, existen formas de mejorar la convergencia de ésta serie.

Aunque el objeto de este post no es obtener la fórmula más eficiente (ver
[artículo](http://numbers.computation.free.fr/Constants/Gamma/gamma.pdf)), sí
que es llamativa la expansión asintótica de la serie, que permite mejorar la
eficiencia enormemente con un simple cambio:

$$
\gamma = lim_{n\to\infty}\left( \sum_{k=1}^\n \frac{1}{k} - \log \left(n+\frac{1}{2}+\frac{1}{24*n}\right)\right)
$$

Esto mejora la convergencia a $\mathcal{O}(\frac{\log n}{3})$.

```julia
function compute_γ(n::Int64)
    harmonic(n) - log(n+0.5+1/(24*n))
end
```

```julia
compute_γ(1_000_000)
```
```bash
0.577215664901(62315)
```

Con esta fórmula calcularemos la constante para la precisión requerida. Para
introducir las capacidades de paralelización de julia se usarán dos estrategias,
una basada en distribución de funciones y otra en la distribución de operaciones
de una sola función.

Si estamos en la REPL de julia (o escribiendo un script), basta con escribir los
siguientes comandos

```julia
using Distributed; addprocs(4)
```

para añadir 4 procesos al entorno de trabajo y cargar las macros que se
necesitan para paralelizar el código (ver
[documentación](https://docs.julialang.org/en/v1/stdlib/Distributed/index.html):
las opciones de la función `addprocs` permiten una configuración del entorno de
paralelización muy avanzada).
Otra opción es iniciar julia directamente con paralelismo:

```bash
$ julia -p 4
```

Adicionalmente, para realizar las medidas de tiempo se usa el paquete
`BenchmarkTools` que define la macro `@btime`, la cual mide el tiempo de
ejecución de la expresión que se le envía varias veces y devuelve el tiempo
medio de ejecución.

```julia
using BenchmarkTools
```

## Estrategia 1: distribución de funciones

Esta primera estrategia trata de definir la función a ser paralelizada sin
ningún tipo de cambio respecto a la programación de un solo proceso (la
"normal", digamos).

Parece útil en casos en los que no esté muy claro dónde se pueden aplicar las
distribuciones de operaciones o si se pretende hacer una prueba rápida de
paralelización sobre un código monoproceso que tuviéramos antes.

En general el modo de empleo es:

* Se añade la macro `@everywhere` delante de la función.
* Se crea un *job* mediante la macro `@spawn`.
* Se hace un `fetch`, que espera a que acabe el *job* y devuelve el valor.

```julia
@everywhere function harmonic_ever(n::Int64)
    x = 0
    for i in n:-1:1
        x +=1/i
    end
    x
end

job = @spawn harmonic_ever(10^10)
fetch(job) # resultado
```

Este tipo de paralelización resulta dificil de medir ya que una vez creado el
`job`, se comienza a ejecutar inmediatamente. Además, `fetch` espera a que acabe
el cálculo para después guardar el resultado en caché y devolverlo. Múltiples
llamadas a `fetch` provocan la devolución de este valor en caché en vez de
repetir el cálculo (que es lo que `@btime` asume).

## Estrategia 2: distribución de operaciones

En este caso se va a paralelizar la operación sumatorio dentro de la función
armónica, es decir, al llamar a ésta función se ejecutará un bucle `for` en
paralelo.
Conviene destacar que la macro utilizada aquí usa *memoria distribuida*, es
decir, funciona también si el programa se ejecuta en un cluster de ordenadores
(por ejemplo, OpenMP tiene una nomenclatura parecida pero sólo funciona en
entornos de memoria compartida).

Para ello, se necesitan unos pequeños cambios:

* La macro `@distributed` se coloca delante del bucle `for` que deba ser
paralelizado y se le indica una operación para hacer la reducción.
* No se inicializa la variable `x` a 0, sino que se iguala al resultado del
bucle paralelizado.
* Entonces, tampoco se hace una asignación dentro del bucle, es decir, se
elimina el `x+=1/i`. Dejar el código así resultaría en un gran número de
asignaciones, muy costosas en memoria y que ralentizarían notablemente el
código.

Así, el nuevo código de la función armónica es:

```julia
function harmonic_dis(n::Int64)
    x = @distributed (+) for i in n:-1:1
        1/i
    end
end
```

Como nota, la operación `+` es una función en julia que no es más especial que
una que pueda definir el usuario, lo que abre un abanico de posibilidades
inmenso (sin embargo, la función debería estar definida como $K^n \to K$ para
evitar bugs).

Los tiempos de cálculo comparados con la función no distribuida son:

```julia
@btime harmonic_dis(10^10)
@btime harmonic(10^10)
```
```bash
12.295 s (410 allocations: 38.50 KiB)
59.068 s (0 allocations: 0 bytes)
```

La mejora de velocidad es lineal, puesto que hay 5 procesos (el original y los 4
añadidos con `addprocs`).

# Resultados

Como paso final, hay que calcular la constante. Como el tipo `double` (o en
julia, `Float64`) sólo tiene capacidad para representar unos 15 decimales, se
usará `BigFloat`, que permite precisión arbitraria.
Para ello, utiliza la librería de GNU [MPFR Library](https://www.mpfr.org/).

Actualmente existen librerías como
[ArbFloat](https://github.com/JuliaArbTypes/ArbFloats.jl) que resultan más
eficientes y rápidas, pero se escapa al objetivo de este post.

Por tanto, se define:

```julia
γ = BigFloat(".57721566490153286060")
```

Sin embargo, hay un problema: si no se define el resultado de la función
`harmonic` como BigFloat, el resultado tendrá desbordamiento aritmético, y por
tanto no será válido. Por tanto, todavía hay que hacer una modificación a las
funciones:

```julia
function harmonic_dis(n::Int64)
    x = @distributed (+) for i in n:-1:1
        1/BigFloat(i)
    end
    x
end

function compute_γ(n::Int64)
    m = BigFloat(n)
    harmonic_dis(n) - log(m+0.5+1/24/m))
end
```

Conviene comentar que hay que prestar atención al uso de `BigFloat`: no se
obtiene el mismo resultado si se usa `BigFloat(1/i)` que si se usa
`1/BigFloat(i)`, debido a que la función `BigFloat` primero ejecuta la expresión
y despues convierte a formato `big`: en el primer caso, esto causa un
desbordamiento en `Float` cuando el denominador es grande, mientras que en el
segundo caso al encontrarse el número `10^10` en formato `BigFloat`, se provoca la
promoción de tipos en todos los elementos involucrados en la operación, y por
tanto ya no hay desbordamiento.

Como comprobación de los 20 dígitos, se ejecuta el siguiente código:

```julia
using Printf
@printf "Calculado: %.21f\nReal:%28.20f\n" compute_γ(10^9) γ
```
```bash
Calculado: 0.577215664901532860605
Real:      0.57721566490153286060
```

Q.E.F.