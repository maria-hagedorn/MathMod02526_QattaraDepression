using GLPK, Cbc, JuMP, SparseArrays, CSV, DataFrames, Ipopt

df = CSV.read("C:/Users/Frederik Danielsen/Documents/Skole/Universitet/DTU/Perioder/Semester 6/Mathematical Modeling/Exercises/Exercise 3 - Quattara Depression/MathMod02526_QattaraDepression/code/points.csv", DataFrame)

# Access the column named "z"
z_coordinates = df[!, :z]

# Optionally, convert the column to an array if you need it in array format
H = Array(z_coordinates)

CHD = 10

K = [[300 140 40],
     [500 230 60],
     [1000 400 70]]

function constructA(H, K)
    A = [abs(j-i) < length(K) ? K[abs(j-i)+1] : 0.0 for i=1:length(H), j=1:length(H)]
    return A
end

function solveIP(H, K)
    h = length(H)
    #myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    myModel = Model(GLPK.Optimizer)
    muModel = Model(Ipopt.Optimizer)

    yields = length(K)

    A = [constructA(H, K[1]), constructA(H, K[2]), constructA(H, K[3])]

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, b[1:yields, 1:h], Bin )  # Choose yield
    @variable(myModel, y[1:h], Int )
    @variable(myModel, u[1:h])  # better version
    @variable(myModel, R[1:h] >= 0 )

    #@objective(myModel, Min, sum(x[j] for j=1:h) )
    @objective(myModel, Min, sum(u[j] for j=1:h) )   # better version

    @constraint(myModel, [j=1:h], R[j] -H[j] -CHD >= -u[j])   # better version
    @constraint(myModel, [j=1:h], u[j] >= R[j] -H[j] -CHD)   # better version
    @constraint(myModel, [j=1:h], R[j] >= H[j] + 10 ) 
    @constraint(myModel, [j=1:h], y[j] >= 1 )  # choose yield
    @constraint(myModel, [j=1:h], 3 >= y[j] )  # choose yield
    @constraint(myModel, [j=1:h-1], 1 >= x[j] + x[j+1])  # not two bombs in a row
    #@constraint(myModel, [i=1:h], R[i] == sum(A[i, j]*x[j] for j=1:h) )
    @constraint(myModel, [i=1:h], R[i] == sum(x[j]*sum(b[k,j]*A[k][i, j] for k=1:yields) for j=1:h) )  # Version wiht yield option
    @constraint(myModel, [j=1:h], sum(b[k,j] for k=1:yields) == 1 )  # Choose yield

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("x = ", JuMP.value.(u))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

solveIP(H,K)
