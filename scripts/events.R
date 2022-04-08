# Sport
addSport <- function(data){
  data$ESport <- sapply(data$age,subSport1)
  data$ERubgy <- data$ESport * sapply(data$sex,subSport2) * sapply(data$nssec5,subSport3)
}data <- data.frame(age = c(10,19,55,87,34,5,23))
sapply(data$age,subSport1)subSport1 <- function(age){
  if(age < 16){
    res <- 0.257
  } else if(age < 25){
    res <- 0.257
  } else if(age < 35){
    res <- 0.233
  } else if(age < 45){
    res <- 0.255
  } else if(age < 55){
    res <- 0.279
  } else if(age < 65){
    res <- 0.268
  } else if(age < 75){
    res <- 0.239
  } else if(age < 85){
    res <- 0.188
  } else {
    res <- 0.090
  }
  return(res)
}
# Prob of <16 set to <25 bc dealing with households is too complicatedsubSport2 <- function(sex){
  res <- 0.63
  if(sex == 0){
    res <- 0.37
  }
  return(res)
}subSport3 <- function(nssec5){
  res <- 1
  if(nssec5 < 4){
    res <- 2
  }
  return(res)
}# Concerts
addConcert <- function(data){
  data$EConcertF <- sapply(data$age,subConcert1)*sapply(data$sex,subConcert3)
  data$EConcertM <- sapply(data$age,subConcert1)*(100 - sapply(data$sex,subConcert3))
  data$EConcertFS <- sapply(data$age,subConcert2)*sapply(data$sex,subConcert3)
  data$EConcertMS <- sapply(data$age,subConcert2)*(100 - sapply(data$sex,subConcert3))
}subConcert1 <- function(age){
  res <- dnorm(age, 23.70431, 5.192425)
  return(res)
}subConcert2 <- function(age){
  res <- dnorm(age, 45.44389, 10.10664)
  return(res)
}subConcert3 <- function(sex){
  res <- 0.30
  if(sex == 0){
    res <- 0.70
  }
  return(res)
}a <- c(8,16.5,19,23,28,32.5,37.5,45.5,55.5,70)
b <- c(0.6,5,13.8,18.2,12.5,7.9,10.8,17.5,10.3,3.4)d <- NULL
for(i in 1:6){
  d <- c(d,rep(a[i],b[i]*10))
}d2 <- NULL
for(i in 6:10){
  d2 <- c(d2,rep(a[i],b[i]*10))
}library(MASS)
fit <- fitdistr(d, "normal")
para <- fit$estimate
fit2 <- fitdistr(d2, "normal")
para2 <- fit2$estimateplot(a,b,xlab = "age",ylab = "prob",pch = 20)
curve(dnorm(x, para[1], para[2])*18.2/dnorm(23, para[1], para[2]), col = 2, add = TRUE)
curve(dnorm(x, para2[1], para2[2])*17.5/dnorm(45.5, para2[1], para2[2]), col = 4, add = TRUE)# Museums
addMuseum <- function(data){
  data$ETankMuseum <- sapply(data$origin,subMuseum2)*sapply(data$nssec5,subMuseum3)
  data$EMuseum <- sapply(data$age,subMuseum1)*data$ETankMuseum
}subMuseum1 <- function(age){
  if(age < 16){
    res <- 0.45
  } else if(age < 25){
    res <- 0.45
  } else if(age < 45){
    res <- 0.54
  } else if(age < 65){
    res <- 0.55
  } else if(age < 75){
    res <- 0.54
  } else {
    res <- 0.36
  }
  return(res)
}
# Prob of <16 set to <25 bc dealing with households is too complicatedsubMuseum2 <- function(origin){
  if(origin == 1){
    res <- 0.53
  } else if(origin == 2){
    res <- 0.63
  } else if(origin == 3){
    res <- 0.46
  } else if(origin == 4){
    res <- 0.28
  } else {
    res <- 0.42
  }
  return(res)
}subMuseum3 <- function(nssec5){
  res<- 0.45
  if(nssec < 8){
    res <- 0.55
  }
  return(res)
}# Religion
addReligion <- function(data){
  data$EReligion1 <- 0
  data$EReligion2 <- 0
  for(i in 1:nrow(data)){
    a = runif(1)
    if(a <= 0.38){
      data$EReligion1 <- 1
    } else if(a <= 0.43){
      data$EReligion2 <- 1
    }
  }
}
