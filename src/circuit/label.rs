use std::{borrow::{Borrow, Cow}, fmt::Display, ops::{Deref, DerefMut}};



#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitLabel(CircuitLabelRef<'static>);
impl CircuitLabel {
    pub fn at_main(label: String) -> Self {
        Self(CircuitLabelRef::at_main_static(label))
    }

    pub fn at_sub_circuit(sub_circuit: String, label: String) -> Self {
        Self(CircuitLabelRef::at_sub_circuit_static(sub_circuit, label))
    }

    /// Maps a None into Some(sub_circuit)
    pub fn map_sub_circuit(self, sub_circuit: String) -> Self {
        Self(self.0.map_sub_circuit_static(sub_circuit))
    }
}
impl<'a> Borrow<CircuitLabelRef<'a>> for CircuitLabel {
    fn borrow(&self) -> &CircuitLabelRef<'a> {
        &self.0
    }
}
impl Clone for CircuitLabel {
    fn clone(&self) -> Self {
        Self(CircuitLabelRef {
            sub_circuit: self.sub_circuit.clone(),
            label: self.label.clone(),
        })
    }
}
impl From<String> for CircuitLabel {
    fn from(value: String) -> Self {
        Self::at_main(value)
    }
}
impl Deref for CircuitLabel {
    type Target = CircuitLabelRef<'static>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for CircuitLabel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Display for CircuitLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(sc) = self.sub_circuit.as_ref() {
            write!(f, "{}::", sc)?;
        }
        write!(f, "{}", self.label)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// `CircuitLabelRef` is to `CircuitLabel` what
/// `&str` is to `String`
///
/// i.e if we have `HashMap<String, T>` we use `.get(key: &str)`
///
/// and if we have `HashMap<CircuitLabel, T>` we can use `.get(key: CircuitLabelRef)`
pub struct CircuitLabelRef<'a> {
    pub(self)sub_circuit: Option<Cow<'a, str>>,
    pub(self)label: Cow<'a, str>,
}
impl CircuitLabelRef<'static> {
    fn at_main_static(label: String) -> Self {
        Self {
            sub_circuit: None,
            label: Cow::Owned(label),
        }
    }

    fn at_sub_circuit_static(sub_circuit: String, label: String) -> Self {
        Self {
            sub_circuit: Some(Cow::Owned(sub_circuit)),
            label: Cow::Owned(label),
        }
    }

    /// Maps a None into Some(sub_circuit)
    fn map_sub_circuit_static(self, sub_circuit: String) -> Self {
        match self {
            Self {
                sub_circuit: None,
                label,
            } => Self {
                sub_circuit: Some(Cow::Owned(sub_circuit)),
                label,
            },
            circuit_label => circuit_label,
        }
    }
}
impl<'a> CircuitLabelRef<'a> {
    pub fn at_main(label: &'a str) -> Self {
        Self {
            sub_circuit: None,
            label: Cow::Borrowed(label),
        }
    }

    pub fn at_sub_circuit(sub_circuit: &'a str, label: &'a str) -> Self {
        Self {
            sub_circuit: Some(Cow::Borrowed(sub_circuit)),
            label: Cow::Borrowed(label),
        }
    }

    pub fn sub_circuit(&self) -> Option<&str> {
        self.sub_circuit.as_ref().map(Deref::deref)
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}
impl<'a> ToOwned for CircuitLabelRef<'a> {
    type Owned = CircuitLabel;

    fn to_owned(&self) -> Self::Owned {
        if let Some(sc) = self.sub_circuit() {
            CircuitLabel::at_sub_circuit(sc.to_owned(), self.label().to_owned())
        } else {
            CircuitLabel::at_main(self.label().to_owned())
        }
    }
}
impl<'a> From<&'a str> for CircuitLabelRef<'a> {
    fn from(value: &'a str) -> Self {
        Self::at_main(value)
    }
}
impl<'a> Display for CircuitLabelRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(sc) = self.sub_circuit.as_ref() {
            write!(f, "{}::", sc)?;
        }
        write!(f, "{}", self.label)
    }
}
