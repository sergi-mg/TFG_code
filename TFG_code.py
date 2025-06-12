#Author: Sergi Martinez Galindo

################################################################################
################################################################################
#1-Simulation and data generation

#initial matrix creation
def initial_matrix(L,rho,proportion):
    """Returns a LxL matrix with {-1,0,1} with a 0's density rho and a
    certain proportion of -1's and 1's, proportion=N+/Ntotal."""
    import random as rand
    import numpy as np
    N_plus=int((proportion)*(1-rho)*L**2)
    N_minus=int((1-proportion)*(1-rho)*L**2)
    matrix=np.zeros((L,L),dtype="float64")

    for k in range(N_plus):
        indexes_zero=np.where(matrix==0)
        i=rand.randint(0,len(indexes_zero[1])-1)
        y=indexes_zero[0][i]
        x=indexes_zero[1][i]
        matrix[y][x]=1

    for k in range(N_minus):
        indexes_zero=np.where(matrix==0)
        i=rand.randint(0,len(indexes_zero[1])-1)
        y=indexes_zero[0][i]
        x=indexes_zero[1][i]
        matrix[y][x]=-1

    return matrix

#System's state evaluation 
def G_function(matrix,threshold,alpha):
    """Given a matrix 2D, with values {-1,0,1}, a threshold between 0 and 1 and
    the value alpha of G, returns a tuple containing a matrix T
    containing the proportion of the other type of agents around, a matrix h
    where happy agents are given a value 1 and the other positions are 0 and
    the value of the function G, global segregation and global happines. Agent
    alone is happy."""
    import numpy as np
    import scipy as sp

    #kernel for convolution
    K=np.ones((3,3),dtype="float64")
    K[1,1]=0

    #occupancy matrix
    O=np.ones(matrix.shape,dtype="float64")-(matrix==0)*1.

    #convolution C1(i)=(N+)-(N-)
    C1=sp.ndimage.convolve(matrix,K,mode="constant",cval=0)*O
    #convolution C2(i)=(N+)+(N-)
    C2=sp.ndimage.convolve(O,K,mode="constant",cval=0)*O

    plus=(C1+C2)/2.
    minus=(C2-C1)/2.

    #only =!0 where matrix!=0

    #we avoid dividing by 0
    eps=10**(-10)
    C2_mod=np.where(C2==0,C2+eps,C2)

    #proportion of different type of agent in first neighbours
    T=(((matrix==1)*1.)*minus+((matrix==-1)*1.)*plus)/C2_mod

    #local happines
    h=(T<=threshold)*1.*O

    #global happines
    if np.sum(O)!=0:
        H=np.sum(h)/np.sum(O)
    else:
        H=np.sum(h)/(10**(-10))


    #segregation
    A=((matrix==1)*1.)*plus+((matrix==-1)*1.)*minus
    if np.sum(C2)!=0:
        S=np.sum(A)/np.sum(C2)
    else:
        S=np.sum(A)/(10**(-10))

    #G function
    G=alpha*S-(1-alpha)*H
    return T,h,G,H,S


#Classic algorithm
def classic_algorithm(M_i,tau,alpha):
    """Executes the classic algorithm and returns a tupple with the final
    configuration, a tupple with (T,h,G,H,S) from the G_function
    applied to the final configuration and the number of iterations.
    The alpha value does not affect the results."""
    import random as rand
    import numpy as np
    #agents matrix
    M=M_i.copy()
    #occupancy matrix
    O=np.ones(M.shape,dtype="float64")-(M==0)*1.
    #total number of empty places
    L=np.size(M_i,0)
    empty_places=L**2-np.sum(O)

    #iterations of the algorithm
    counter=0 #will count the number of movements applied
    iterations=0 #will count the total number of iterations
    N_uh=10 #number !=0 to start the loop
    while N_uh!=0:
        results=G_function(M, tau, alpha)

        #matrix that shows where the unhappy agents are
        T=results[0]
        U=O*(T>tau)
        #total number of unhappy agents
        N_uh=np.sum(U)

        U_mod=U.copy()

        #now we have to find a suitable movement randomly
        movement=0
        change_r_a=0 #controls the total number of agents tried to be moved

        while movement==0:
            change_r_a+=1
            N_uh=np.sum(U_mod)
            if N_uh==0:
                break

            #we have to choose an unhappy agent randomly

            indexes_unhappy=np.where(U_mod==1)
            i=rand.randint(0,len(indexes_unhappy[1])-1)


            uh_y=indexes_unhappy[0][i]
            uh_x=indexes_unhappy[1][i]

            #location change selection
            O_mod=O.copy()
            #we have to choose an empty position randomly
            it=0
            finish=0
            while finish==0:
                #empty position selection
                indexes_empty=np.where(O_mod==0)
                j=rand.randint(0,len(indexes_empty[1])-1)

                e_y=indexes_empty[0][j]
                e_x=indexes_empty[1][j]

                #we check if the agent would be happy in the new postion
                M_mod=M.copy()
                M_mod[uh_y,uh_x]=0
                M_mod[e_y,e_x]=M[uh_y,uh_x]
                modified=G_function(M_mod, tau, alpha)


                #if the agent would be happy we make the change
                if modified[1][e_y,e_x]==1:
                    M=M_mod.copy()
                    O=np.ones(M.shape,dtype="float64")-(M==0)*1.
                    counter+=1
                    finish=1
                    movement=1
                #if not we look for another random empty place for this agent
                else:
                    O_mod[e_y,e_x]=1
                    it+=1
                #if this agent cannot move
                #we look for another random unhappy agent (different iteration)
                if it==(empty_places):
                    U_mod[uh_y,uh_x]=0
                    break


        iterations+=1
        #control sequence to avoid an infinite loop
        if iterations>100000:
            break

    return(M,results,iterations)

#Greedy algorithm
def our_model(M_i, tau, alpha):
    """Executes the greedy algorithm and returns a tupple with the final
    configuration, a tupple with (T,h,G,H,S) from the G_function
    applied to the final configuration and the number of iterations."""
    import random as rand
    import numpy as np

    #agents matrix
    M=M_i.copy()
    #occupancy matrix
    O=np.ones(M.shape,dtype="float64")-(M==0)*1.

    #iterations of the algorithm
    counter=0 #will count the number of movements applied
    iterations=0 #will count the total number of iterations
    N_uh=10 #number !=0 to start the loop
    while N_uh!=0:
        results=G_function(M, tau, alpha)
        #if there are no empty places, we stop
        if np.sum(O)==np.size(M):
            break

        #we have to choose an unhappy agent randomly

        #matrix that shows where the unhappy agents are
        T=results[0]
        U=O*(T>tau)
        #total number of unhappy agents
        N_uh=np.sum(U)
        #print(iterations,N_uh)

        U_mod=U.copy()

        #looking for a suitable movement
        movement=0
        change_r_a=0 #controls the total number of agents tried to be moved
        while movement==0:
            change_r_a+=1
            N_uh=np.sum(U_mod)
            if N_uh==0:
                #print("agents tried:",str(change_r_a))
                break

            #choosing a random unhappy agent

            indexes_unhappy=np.where(U_mod==1)
            i=rand.randint(0,len(indexes_unhappy[1])-1)


            uh_y=indexes_unhappy[0][i]
            uh_x=indexes_unhappy[1][i]

            #empty locations
            indexes_empty=np.where(M==0)

            #we look the location which will give the minimum free energy
            G_list=np.zeros((len(indexes_empty[1])),dtype="float64")
            loc_h_list=np.zeros((len(indexes_empty[1])),dtype="float64")
            for j in range(len(indexes_empty[1])):
                #empty location selection
                e_y=indexes_empty[0][j]
                e_x=indexes_empty[1][j]
                #new matrix
                M_mod=M.copy()
                M_mod[uh_y,uh_x]=0
                M_mod[e_y,e_x]=M[uh_y,uh_x]
                modified_results=G_function(M_mod, tau, alpha)
                #free energy results for each location
                G_list[j]=modified_results[2]
                loc_h_list[j]=modified_results[1][e_y,e_x]

            #movement selection
            finish=0
            it=0
            while finish==0:
                #minimum free energy value:
                #position minG_new = postion min(G_new-G_old) as G_old=ctt
                minimum=np.argmin(G_list)
                #if conditions are met we make the movement
                if (loc_h_list[minimum]==1):
                    #empty location selection
                    e_y=indexes_empty[0][minimum]
                    e_x=indexes_empty[1][minimum]
                    #new matrix
                    M[e_y,e_x]=M[uh_y,uh_x]
                    M[uh_y,uh_x]=0
                    O=np.ones(M.shape,dtype="float64")-(M==0)*1.
                    counter+=1
                    finish=1
                    movement=1
                #if not, we look for the following empty place
                else:
                    G_list[minimum]=10 #>G always
                    it+=1
                #if no empty place is suitable, we go to another unhappy agent
                if it==len(indexes_empty[1]):
                    U_mod[uh_y,uh_x]=0
                    break

        iterations+=1
        #control sequence to avoid an infinite loop
        if iterations>100000:
            break

    return(M,results,iterations)

#data generation
def data_generator(alpha,rho_0,N):
    import numpy as np
    """Executes N simulations of the algorithm and then writes a data
    file with three columns: final S, H and number of steps, in that order."""
    tau=0.5 #in this study, it can be changed for new studies
    data=np.zeros((N,3),dtype="float64")
    file_name="alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_N_"+\
    str(N)+"_2.dat"
    for i in range(N):
        M_i=initial_matrix(20, rho_0, 0.5)
        #the size L=20 an proportion 0.5 can be changed for other studies
        #simulation execution (to execute the classic algorithm
        #change "our_model" for "classic_algorithm".)
        final_state=our_model(M_i, tau, alpha)
        data[i][0]=final_state[1][4]
        data[i][1]=final_state[1][3]
        data[i][2]=final_state[2]
    np.savetxt(file_name, data, fmt="%.16e", delimiter="   ")

################################################################################
################################################################################
#2-Data analysis

def statistics(x):
    """Returns the mean and its sigma, you can get the uncertainty with
    95 % of confidence with 1.96*sigma."""
    import numpy as np
    sumx=np.sum(x)
    sumx2=np.sum(x**2)
    N=np.size(x)
    xmed=sumx/N
    x2med=sumx2/N
    #distribution's statistics
    s2=(N*(x2med-xmed**2))/(N-1)
    s=s2**0.5
    #mean value's statistics
    var=s2/N
    if var<0:
        var=0
    sigma=(var)**0.5
    return xmed,sigma

def xifres(values,uncertainties,exp_max,exp_min):
    import numpy as np
    """Rounds the values and their uncertainties with the
    adecuate number of significant figures. Values and uncertanties
    are 1D matrices and exp_max and exp_min are the power (x) of 10^x
    corresponding to the maximum and minimum order of the uncertainties,
    if you get 0's in the output try exppanding the exponent range."""
    values_c=np.zeros_like(values)
    uncertainties_c=np.zeros_like(uncertainties)
    for i in range(np.size(uncertainties)):
        for k in range(exp_max,exp_min,-1):
            if uncertainties[i]==0:
                values_c[i]=values[i]
                uncertainties_c[i]=uncertainties[i]
            if uncertainties[i]>=1.95*10**k:
                values_c[i]=round(values[i],-k)
                uncertainties_c[i]=round(uncertainties[i],-k)
                break
    return values_c,uncertainties_c

def calc_reg(x,y):
    import numpy as np
    """Returns the parameters of a linear regression of y(x).
    y and x are numpy 1D-arrays."""
    #y=mx+b
    N=np.size(x)
    #mean values
    xmed=np.sum(x)/N
    x2med=np.sum(x**2)/N
    ymed=np.sum(y)/N
    y2med=np.sum(y**2)/N
    xymed=np.sum(x*y)/N
    #sigma's
    sigx2=x2med-xmed**2
    sigy2=y2med-ymed**2
    sigxy=xymed-xmed*ymed
    #regression parameters
    m=sigxy/sigx2
    b=ymed-m*xmed
    r=sigxy/((sigx2*sigy2)**0.5)
    #uncertainties
    dyreg=((sigy2*(1-r**2)*N)/(N-2))**0.5
    dm=dyreg/(N*sigx2)**0.5
    db=dyreg*(x2med/(N*sigx2))**0.5

    return m,dm,b,db,r

def statistics_matrix(rho_val,alpha_val,N,number):
    """Reads the files for the rho and alpha (1D arrays) indicated and returns
    the mean and its sigma of the three variables S,H,steps for each
    pair rho-alpha: M[alpha,rho,variable] for mean and sigma, two components
    of the tuple. It also returns the correlation coefficient r of S(steps)
    for each pair (alpha,rho)."""
    import numpy as np
    N_alpha=np.size(alpha_val)
    N_rho=np.size(rho_val)
    mean=np.zeros((N_alpha,N_rho,3),dtype="float64")
    sigma=np.zeros((N_alpha,N_rho,3),dtype="float64")
    correlation=np.zeros((N_alpha,N_rho),dtype="float64")
    for i in range(N_alpha):
        alpha=alpha_val[i]
        for j in range(N_rho):
            rho=rho_val[j]
            #reading the file
            file_name="alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho,2))\
                +"_N_"+str(N)+"_2".dat"
            read_M=np.loadtxt(fname=file_name,dtype="float64")
            #variables studied
            S=read_M[:,0]
            H=read_M[:,1]
            steps=read_M[:,2]
            #steps-segregation correlation
            (m,dm,b,db,r)=calc_reg(S, steps)
            correlation[i,j]=r
            #mean and sigma calculus
            S_stats=statistics(S)
            H_stats=statistics(H)
            steps_stats=statistics(steps)
            #saving the results
            mean[i,j,0]=S_stats[0]
            mean[i,j,1]=H_stats[0]
            mean[i,j,2]=steps_stats[0]

            sigma[i,j,0]=S_stats[1]
            sigma[i,j,1]=H_stats[1]
            sigma[i,j,2]=steps_stats[1]

    return mean,sigma,correlation


#we can obtain the results needed with the following code
#(previously all the files for each pair must have been created, if
#you have other values of rho and alpha, modify the arrays alpha_l and rho_l)
import numpy as np
alpha_l=np.zeros((37))
alpha_l[0]=0
alpha_l[1]=0.0001
alpha_l[2]=0.005
alpha_l[3:5]=np.arange(0.01,0.021,0.01)
alpha_l[5:13]=np.arange(0.05,0.42,step=0.05)
alpha_l[13:]=np.arange(0.425,1.01,step=0.025)

rho_l=np.zeros((30))
rho_l[:14]=np.arange(0.01,0.15,0.01)
rho_l[14:]=np.arange(0.15,0.92,step=0.05)

A=statistics_matrix(rho_l, alpha_l, 100)

results=np.zeros((37,30,3))
d_results=np.zeros((37,30,3))
for v in range (3):
    for i in range(np.size(rho_l)):
        results[:,i,v],d_results[:,i,v]=xifres(A[0][:,i,v],\
                                               1.96*A[1][:,i,v],10,-10)

################################################################################
################################################################################
#3-Other data generation
#Initial segregation
import numpy as np
rho_l=np.zeros((31))
rho_l[:16]=np.arange(0.01,0.161,0.01)
rho_l[16:]=np.arange(0.2,0.92,step=0.05)
N=10000 
S_rho=np.zeros((N,31),dtype="float64")
for rho_i in range(0,31):
    rho=rho_l[rho_i]
    print(rho)    
    for i in range(N):
        M=initial_matrix(20, rho, 0.5)
        results=free_energy(M, 0.5, 0.5) #the alpha value is not important here
        S_rho[i][rho_i]=results[4]
#saving the data
np.savetxt("s_ini_rho_10000.dat", S_rho, fmt="%.16e", delimiter="   ")

#Initial number of unhappy agents (simulation)
import numpy as np
N=10000
ua_rho=np.zeros((N,39),dtype="float64")
import numpy as np
for rho_i in range(0,39):
    rho=rho_i/40+0.025
    print(rho)    
    for i in range(N):
        M=initial_matrix(20, rho, 0.5)
        O=np.ones(M.shape,dtype="float64")-(M==0)*1.
        results=free_energy(M, 0.5, 0.5) #the alpha value is not important here
        T_i=results[0]
        U=O*(T_i>0.5)
        #total number of unhappy agents
        N_uh=np.sum(U)
        ua_rho[i][rho_i]=N_uh
#saving the data
np.savetxt("ua_rho_10000.dat", ua_rho, fmt="%.16e", delimiter="   ")

#Initial number of unhappy agents (theoretical)
import numpy as np
import scipy.special as sp
N=400
rho_0=np.arange(0.025,0.976,0.025)
Pi=(N*(1-rho_0)/2-1)/(N-1) #equal
Pd=(N*(1-rho_0)/2)/(N-1) #different
P0=rho_0*N/(N-1) #agent-vacancy
#expected value for individual happiness <h>=h/norm
h=0 
norm=0 
for Ni in range(8+1):
    for Nd in range(8-Ni+1):
        N0=8-Ni-Nd
        norm+=(Pi**Ni)*(Pd**Nd)*(P0**N0)*\
        (sp.factorial(8)/(sp.factorial(Ni)*sp.factorial(Nd)*sp.factorial(N0)))
        if Nd<=Ni:
            #the agent will be happy
            h+=(Pi**Ni)*(Pd**Nd)*(P0**N0)*\
            (sp.factorial(8)/(sp.factorial(Ni)*sp.factorial(Nd)*sp.factorial(N0)))
#number of unhappy agents: ua
ua=N*(1-rho_0)*(1-h/norm)
rho_plot=np.arange(0.025,1,step=0.025)
#saving the data
save_U=np.zeros((39,2),dtype="float64")
save_U[:,0]=rho_plot
save_U[:,1]=ua.copy()
np.savetxt("Ua_results_theoric.dat", save_U, delimiter="   ", fmt="%.10f")
