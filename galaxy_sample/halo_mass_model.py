import numpy as np
from sklearn import linear_model, preprocessing
from joblib import load, dump
import my_functions as mf

class HaloMassModel:
    
    def __init__(self,gf):
        self.parent = gf
        
        src = '/scratch/Documents/Conformity2019/SLH2020/models/'
        ScalerParams = load(src+'ScalerParams')
        PolyParams = load(src+'PolyParams')
        self.Model = load(src+'Model')
        
        Scaler = preprocessing.StandardScaler()
        Scaler.mean_, Scaler.var_, Scaler.scale_ = ScalerParams
        Poly = preprocessing.PolynomialFeatures(PolyParams)
        
        self.Scaler, self.Poly = Scaler, Poly
        
    def GetFeatures(self,group):
        
        LogGroupLen = lambda members: np.log10(len(members))
        MaxMass = lambda members: np.max(self.parent.sample.data['StellarMass'][members])
        TotMass = lambda members: np.sum(self.parent.sample.data['StellarMass'][members])
        Zvar = lambda members: np.var(self.parent.sample.data['redshift'][members])
        GroupRedshift = lambda members: np.mean(self.parent.sample.data['redshift'][members])
        
        X = np.array([f(group.members) for f in [LogGroupLen,MaxMass,TotMass,Zvar,GroupRedshift]]).reshape(1,-1)
        X = self.Poly.fit_transform(self.Scaler.transform(X))
        
        return X
    
    def Predict(self,group):
        return self.Model.predict( self.GetFeatures(group) )[0]
    
class SimpleModel:
    
    def __init__( self, src=None, load_params=True):
        if load_params:
            self.Model = load(src+'Model')
        else: 
            self.Model = linear_model.LinearRegression()
            
    def get_features(self,sample,groups,training=False):
        TotMass = lambda sample,members: np.log10(np.sum(10**sample.data['StellarMass'][members]))
        GroupMass = lambda sample,members: sample.data['CentralMvir'][members][0]

        GetFeature = lambda f: [f( sample, group ) for group in groups.values()]
        features = np.array([GetFeature(TotMass),np.array(GetFeature(TotMass))**2]).T.reshape(-1,2)
        
        if training:
            target = np.array(GetFeature(GroupMass))
        
        return (features,target) if training else features

    def train(self,sample,groups):
        features, target = self.get_features(sample,groups,training=True)
        self.Model.fit( features, target )
        
    def dump_params(self,dst):
        dump( self.Model, dst+'Model')
           
    def predict(self,sample,groups):
        """
        sample: galaxy_sample instance
        groups: groups in dict form
        """
        features = self.get_features(sample,groups)
        prediction = self.Model.predict( features )

        return prediction    
    
class Interpolator:
    
    def __init__(self,xp,yp):
        I = np.argsort(xp)
        self.xp,self.yp = xp[I],yp[I]
        
    def get_features(self,sample,groups):
        TotMass = lambda sample,members: np.log10(np.sum(10**sample.data['StellarMass'][members]))
        GetFeature = lambda f: [f( sample, group ) for group in groups.values()]
        totalmass = np.array(GetFeature(TotMass))
        return totalmass
        
    def predict(self,sample,groups):
        x = self.get_features(sample,groups)
        return np.interp(x,self.xp,self.yp)
    
class AbundanceMatching:
    
    def __init__(self,src=None, load_pop=True, SHAM=False):
        self.SHAM=SHAM
        fname = 'HaloMassFunction' if not self.SHAM else 'SubHaloMassFunction'
        if load_pop:
            self.population = load(src+fname)
            
    def define_population(self,sample):
        if self.SHAM:
            self.population = sample.data['Mvir']
        else:
            TrueGroups = dict()
            for idx,GrNr in enumerate(sample.data['FOFCentralGal']):
                if GrNr not in TrueGroups:
                    TrueGroups[GrNr] = [idx]
                else:
                    TrueGroups[GrNr].append(idx)

            self.population = self.get_features(sample,TrueGroups,training=True)
        
    def dump_population(self,dst):
        fname = 'HaloMassFunction' if not self.SHAM else 'SubHaloMassFunction'
        dump( self.population, dst+fname )
            
    def get_features(self,sample,groups,training=False):
        GroupMass = lambda sample,members: sample.data['CentralMvir'][members][0]
        TotMass = lambda sample,members: np.log10(np.sum(10**sample.data['StellarMass'][members]))

        GetFeature = lambda f: [f( sample, group ) for group in groups.values()]
        
        groupmass = np.array(GetFeature(GroupMass))
        totalmass = np.array(GetFeature(TotMass))
        
        return groupmass if training else totalmass
    
    def draw(self,count):
        return np.sort(np.random.choice(self.population,count))
        
    def predict(self,sample,groups):
        """
        sample: galaxy_sample instance
        groups: groups in dict form. Ignored in the case of SHAM
        """
        
        if self.SHAM:
            I = np.argsort(sample.data['StellarMass'])
            order = np.argsort(np.arange(sample.count)[I])
            TheoreticalMasses = self.draw(sample.count)
        else:
            TotMass = self.get_features(sample,groups)
            I = np.argsort(TotMass)
            order = np.argsort(np.arange(len(groups))[I])
            TheoreticalMasses = self.draw(len(groups))
            
        prediction = TheoreticalMasses[order]
        return prediction
    
    
class RedshiftDependentAbundanceMatching:
    """
    Does abundance matching while taking into account redshift-dependent completeness limit
    """
    def __init__( self, redshift_bins, src=None, load_pop=True, SHAM=False):
        self.SHAM=SHAM
        self.redshift_bins = redshift_bins
        fname = 'HaloMassFunction' if not self.SHAM else 'SubHaloMassFunction'
        if load_pop:
            self.population = load(src+fname)
            
    def define_population(self,sample):
        
        self.population = dict()
        
        if self.SHAM:
            for z_low,z_high,members in self.iter_zbins(sample.data['redshift']):
                self.population[(z_low,z_high)] = sample.data['Mvir'][members]
        else:
            TrueGroups = dict()
            for idx,GrNr in enumerate(sample.data['FOFCentralGal']):
                if GrNr not in TrueGroups:
                    TrueGroups[GrNr] = [idx]
                else:
                    TrueGroups[GrNr].append(idx)

            groupmass,redshift = self.get_features(sample,TrueGroups,training=True)
            for z_low,z_high,members in self.iter_zbins(redshift):
                self.population[(z_low,z_high)] = groupmass[members]
        
    def dump_population(self,dst):
        fname = 'HaloMassFunction' if not self.SHAM else 'SubHaloMassFunction'
        dump( self.population, dst+fname )
    
    def draw(self,zbin,count):
        population = self.population[zbin]
        return np.sort(np.random.choice(population,count))
            
    def get_features(self,sample,groups,training=False):
        GroupMass = lambda sample,members: sample.data['CentralMvir'][members][0]
        TotMass = lambda sample,members: np.log10(np.sum(10**sample.data['StellarMass'][members]))
        GroupRedshift = lambda sample,members: np.mean(sample.data['redshift'][members])
        GetFeature = lambda f: np.array([f( sample, group ) for group in groups.values()])
        if training:
            return (GetFeature(GroupMass),GetFeature(GroupRedshift))
        else:
            return (GetFeature(TotMass),GetFeature(GroupRedshift))
        
    def iter_zbins(self,redshift):
        for (z_low,z_high),(_,members) in zip(zip(self.redshift_bins[:-1],self.redshift_bins[1:]),
                                              mf.binmembers(redshift, self.redshift_bins)   ):
            yield z_low,z_high,members

    def predict(self,sample,groups):
        """
        sample: galaxy_sample instance
        groups: groups in dict form
        """
        prediction = np.zeros(len(groups),)
        
        if self.SHAM:
            for z_low,z_high,members in self.iter_zbins(sample.data['redshift']):

                I = np.argsort(sample.data['StellarMass'][members])
                order = np.argsort(np.arange(len(members))[I])

                TheoreticalMasses = self.draw((z_low,z_high),len(members))
                prediction[members] = TheoreticalMasses[order]
                
        else:
            totalmass, redshift = self.get_features(sample,groups)
            for z_low,z_high,members in self.iter_zbins(redshift):

                TotMass = totalmass[members]

                I = np.argsort(TotMass)
                order = np.argsort(np.arange(len(members))[I])

                TheoreticalMasses = self.draw((z_low,z_high),len(members))
                prediction[members] = TheoreticalMasses[order]
            
        return prediction
    
    
class RedshiftDependentModel:
    
    def __init__( self, src=None, load_params=True, redshift_bins=np.arange(0.02,0.09,0.01).astype('float32')):
        
        self.redshift_bins = redshift_bins
        
        self.model = linear_model.RidgeCV
        self.PolyParam = 2
        
        if load_params:
            self.Models = load(src+'Models')
            self.ScalerParams = load(src+'ScalerParams')
        else: 
            self.Models = dict()
            self.ScalerParams = dict()
            
    def get_features(self,sample,groups,training=False):
        
        LogGroupLen = lambda sample,members: np.log10(len(members))
        MaxMass = lambda sample,members: np.max(sample.data['StellarMass'][members])
        TotMass = lambda sample,members: np.log10(np.sum(10**sample.data['StellarMass'][members]))
        Zvar = lambda sample,members: np.var(sample.data['redshift'][members])
        GroupRedshift = lambda sample,members: np.mean(sample.data['redshift'][members])
        
        GroupMass = lambda sample,members: sample.data['CentralMvir'][members][0]

        GetFeature = lambda f: [f( sample, group ) for group in groups.values()]
        
        features = np.array(list(map(GetFeature, [LogGroupLen,MaxMass,TotMass,Zvar]))).T
        redshift = np.array(GetFeature(GroupRedshift))
        
        if training:
            target = np.array(GetFeature(GroupMass))
        
        return (features,redshift,target) if training else (features,redshift)
        
        
    def iter_zbins(self,redshift):
        for (z_low,z_high),(_,members) in zip(zip(self.redshift_bins[:-1],self.redshift_bins[1:]),
                                              mf.binmembers(redshift, self.redshift_bins)   ):
            yield z_low,z_high,members
                
                
    def preprocess(self,x,z_low,z_high,training=False):

        Poly = preprocessing.PolynomialFeatures(self.PolyParam)
        Scaler = preprocessing.MinMaxScaler()

        if training:
            X = Poly.fit_transform(Scaler.fit_transform(x))
            self.ScalerParams[(z_low,z_high)] = (Scaler.data_min_, Scaler.data_max_, Scaler.scale_)
        else:
            Scaler.data_min_, Scaler.data_max_, Scaler.scale_ = self.ScalerParams[(z_low,z_high)]
            Scaler.min_ = Scaler.feature_range[0] - Scaler.data_min_ * Scaler.scale_
            
            X = Poly.fit_transform(Scaler.transform(x))
            
        return X
        
        
    def train(self,sample,groups):
        
        features, redshift, target = self.get_features(sample,groups,training=True)
        
        for z_low,z_high,members in self.iter_zbins(redshift):
            
            X = self.preprocess( features[members,:], z_low, z_high, training=True )
            y = target[members]
            
            self.Models[(z_low,z_high)] = self.model().fit( X, y )
        
        
    def dump_params(self,dst):
        dump( self.Models, dst+'Models')
        dump( self.ScalerParams, dst+'ScalerParams')
        
        
    def predict(self,sample,groups):
        """
        sample: galaxy_sample instance
        groups: groups in dict form
        """
        features, redshift = self.get_features(sample,groups)
        
        prediction = np.zeros(len(features),)
        for z_low,z_high,members in self.iter_zbins(redshift):
            if len(members)==0:
                pass
            else:
                model = self.Models[(z_low,z_high)]
                X = self.preprocess( features[members,:], z_low, z_high )
                prediction[members] = model.predict( X )

        return prediction